use libc::{c_uint, size_t};
use std::convert::TryInto;
use std::ffi::CString;
#[cfg(windows)]
use std::ffi::OsStr;
#[cfg(unix)]
use std::os::unix::ffi::OsStrExt;
use std::path::{Path, PathBuf};
use std::sync::atomic::AtomicU32;
use std::sync::Mutex;
use std::{fmt, mem, ptr, result};

use lmdb_sys as ffi;

use byteorder::{ByteOrder, NativeEndian};

use crate::cursor::Cursor;
use crate::database::Database;
use crate::error::{lmdb_result, Error, Result};
use crate::flags::{DatabaseFlags, EnvironmentFlags};
use crate::transaction::{RoTransaction, RwTransaction, Transaction};

use crate::resize::{DatabaseResizeInfo, DatabaseResizeSettings, DEFAULT_RESIZE_SETTINGS};
use crate::transaction_guard::{ScopedTransactionBlocker, TransactionGuard};

#[cfg(windows)]
/// Adding a 'missing' trait from windows OsStrExt
trait OsStrExtLmdb {
    fn as_bytes(&self) -> &[u8];
}
#[cfg(windows)]
impl OsStrExtLmdb for OsStr {
    fn as_bytes(&self) -> &[u8] {
        &self.to_str().unwrap().as_bytes()
    }
}

/// An LMDB environment.
///
/// An environment supports multiple databases, all residing in the same shared-memory map.
pub struct Environment {
    env: *mut ffi::MDB_env,
    tx_count: AtomicU32,
    tx_blocker_count: AtomicU32,
    db_resize_lock: Mutex<()>,
    dbi_open_mutex: Mutex<()>,
    // - Send + Sync is required because Environment asserts both, and the callback is carried
    //   along with it; without the bound a closure could capture a value that is unsafe to
    //   share across threads, and the unsafe impls below would be handing out a false promise
    resize_callback: Option<Box<dyn Fn(DatabaseResizeInfo) + Send + Sync>>,
    resize_settings: Option<DatabaseResizeSettings>,
    db_path: PathBuf,
}

impl Environment {
    /// Creates a new builder for specifying options for opening an LMDB environment.
    #[allow(clippy::new_ret_no_self)]
    pub fn new() -> EnvironmentBuilder {
        EnvironmentBuilder {
            flags: EnvironmentFlags::empty(),
            max_readers: None,
            max_dbs: None,
            map_size: None,
            resize_callback: None,
            resize_settings: None,
        }
    }

    /// Returns a raw pointer to the underlying LMDB environment.
    ///
    /// The caller **must** ensure that the pointer is not dereferenced after the lifetime of the
    /// environment.
    pub fn env(&self) -> *mut ffi::MDB_env {
        self.env
    }

    /// Opens a handle to an LMDB database.
    ///
    /// If `name` is `None`, then the returned handle will be for the default database.
    ///
    /// If `name` is not `None`, then the returned handle will be for a named database. In this
    /// case the environment must be configured to allow named databases through
    /// `EnvironmentBuilder::set_max_dbs`.
    ///
    /// The returned database handle may be shared among any transaction in the environment.
    ///
    /// This function will fail with `Error::BadRslot` if called by a thread which has an ongoing
    /// transaction.
    ///
    /// The database name may not contain the null character.
    pub fn open_db(&self, name: Option<&str>) -> Result<Database> {
        let mutex = self.dbi_open_mutex.lock();
        let txn = self.begin_ro_txn()?;
        let db = unsafe { txn.open_db(name)? };
        txn.commit()?;
        drop(mutex);
        Ok(db)
    }

    /// Opens a handle to an LMDB database, creating the database if necessary.
    ///
    /// If the database is already created, the given option flags will be added to it.
    ///
    /// If `name` is `None`, then the returned handle will be for the default database.
    ///
    /// If `name` is not `None`, then the returned handle will be for a named database. In this
    /// case the environment must be configured to allow named databases through
    /// `EnvironmentBuilder::set_max_dbs`.
    ///
    /// The returned database handle may be shared among any transaction in the environment.
    ///
    /// This function will fail with `Error::BadRslot` if called by a thread with an open
    /// transaction.
    pub fn create_db(&self, name: Option<&str>, flags: DatabaseFlags) -> Result<Database> {
        let mutex = self.dbi_open_mutex.lock();
        let txn = self.begin_rw_txn(None)?;
        let db = unsafe { txn.create_db(name, flags)? };
        txn.commit()?;
        drop(mutex);
        Ok(db)
    }

    /// Retrieves the set of flags which the database is opened with.
    ///
    /// The database must belong to to this environment.
    pub fn get_db_flags(&self, db: Database) -> Result<DatabaseFlags> {
        let txn = self.begin_ro_txn()?;
        let mut flags: c_uint = 0;
        unsafe {
            lmdb_result(ffi::mdb_dbi_flags(txn.txn(), db.dbi(), &mut flags))?;
        }
        Ok(DatabaseFlags::from_bits(flags).expect("lmdb: Database Flags that are expected to work, failed"))
    }

    /// Create a read-only transaction for use with the environment.
    pub fn begin_ro_txn(&self) -> Result<RoTransaction<'_>> {
        RoTransaction::new(self)
    }

    fn headroom_from_ratio(current_map_size: usize, resize_ratio: u32) -> usize {
        ((current_map_size as u128 * resize_ratio as u128) / 100).try_into().expect("lmdb: Failed to convert headroom value to usize; this means either database configuration is wrong or an invariant is broken")
    }

    /// The caller must hold `db_resize_lock`; the only caller, `begin_rw_txn_generic`, holds it,
    /// so the resize it drives through `do_resize_locked` shares that single-owner critical section.
    fn resize_db_if_necessary(&self, headroom: Option<usize>) -> Result<()> {
        let resize_settings = self.resize_settings.as_ref().unwrap_or(&DEFAULT_RESIZE_SETTINGS);

        let env_info = self.info().expect("Environment info retrieval failed while resizing");
        let initial_map_size = env_info.map_size();

        let required_space = headroom.unwrap_or_else(|| {
            Self::headroom_from_ratio(initial_map_size, resize_settings.default_resize_ratio_percentage)
        });
        while self.needs_resize(headroom)? {
            let new_map_size = self.do_resize_locked(Some(required_space))?;
            if new_map_size >= required_space + initial_map_size {
                break;
            }
        }
        Ok(())
    }

    /// Create a read-write transaction for use with the environment. This method will block while
    /// there are any other read-write transactions open on the environment.
    pub fn begin_rw_txn(&self, headroom: Option<usize>) -> Result<RwTransaction<'_>> {
        self.begin_rw_txn_generic(headroom, false, false)
    }

    /// Create a read-write transaction for use with the environment. This method will block while
    /// there are any other read-write transactions open on the environment.
    ///
    /// This is a more generic version of `begin_rw_txn`, which allows to optionally pass the MDB_NOSYNC
    /// or MDB_NOMETASYNC flags to the underlying call to `mdb_txn_begin`. Passing the flags here has
    /// the same effect as passing them when opening the environment, but only this particular transaction
    /// will be affected. Same caveats apply:
    /// 1) If MDB_NOSYNC is passed, a system crash may undo an already committed tx or corrupt the db,
    ///    depending on the underlying filesystem (the corruption is possible e.g. on ext4 in "writeback"
    ///    mode, on NTFS, probably on APFS too).
    /// 2) If MDB_NOMETASYNC is passed, a system crash may undo an already committed tx.
    ///
    /// Note that the flags from the environment are just ORed with those passed to `mdb_txn_begin`,
    /// so you can't "undo" a flag from the environment by passing false for one of the parameters here.
    pub fn begin_rw_txn_generic(
        &self,
        headroom: Option<usize>,
        no_sync: bool,
        no_meta_sync: bool,
    ) -> Result<RwTransaction<'_>> {
        // - the expect covers poisoning only, which means another thread panicked mid-resize and left
        //   the map in an unknown state: a broken invariant, not a runtime condition to hand back
        // - returning it instead is not expressible: Error carries an OS errno and has no poison
        //   variant, so propagating would mean inventing an errno the OS never reported
        let _lock = self.db_resize_lock.lock().expect("Database resize mutex lock failed");
        self.resize_db_if_necessary(headroom)?;
        RwTransaction::new(self, no_sync, no_meta_sync)
    }

    /// Flush data buffers to disk.
    ///
    /// Data is always written to disk when `Transaction::commit` is called, but the operating
    /// system may keep it buffered. LMDB always flushes the OS buffers upon commit as well, unless
    /// `MDB_NOSYNC` or `MDB_NOMETASYNC` were passed when opening the environment or creating the
    /// transaction.
    ///
    /// Note:
    /// * If the environment was opened with `MDB_NOSYNC`, `sync` will do nothing unless
    ///   `force` is set to true.
    /// * It will effectively "fix" the potential consistency issues introduced by previous
    ///   `MDB_NOSYNC` commits (by ensuring that all transaction data has been written to disk).
    /// * It is independent from the transactions machinery and can be called concurrently
    ///   with transaction creation or committing or with itself.
    pub fn sync(&self, force: bool) -> Result<()> {
        unsafe { lmdb_result(ffi::mdb_env_sync(self.env(), i32::from(force))) }
    }

    /// Return the number of transactions currently running; controlled with TransactionGuard objects
    pub(crate) fn tx_count(&self) -> &AtomicU32 {
        &self.tx_count
    }

    /// Return the number of requests to block any new transactions, controlled with ScopedTransactionBlocker
    pub(crate) fn tx_blocker_count(&self) -> &AtomicU32 {
        &self.tx_blocker_count
    }

    /// Closes the database handle. Normally unnecessary.
    ///
    /// Closing a database handle is not necessary, but lets `Transaction::open_database` reuse the
    /// handle value. Usually it's better to set a bigger `EnvironmentBuilder::set_max_dbs`, unless
    /// that value would be large.
    ///
    /// ## Safety
    ///
    /// This call is not mutex protected. Databases should only be closed by a single thread, and
    /// only if no other threads are going to reference the database handle or one of its cursors
    /// any further. Do not close a handle if an existing transaction has modified its database.
    /// Doing so can cause misbehavior from database corruption to errors like
    /// `Error::BadValSize` (since the DB name is gone).
    pub unsafe fn close_db(&mut self, db: Database) {
        ffi::mdb_dbi_close(self.env, db.dbi());
    }

    /// Retrieves statistics about this environment.
    pub fn stat(&self) -> Result<Stat> {
        unsafe {
            let mut stat = Stat::new();
            lmdb_try!(ffi::mdb_env_stat(self.env(), stat.mdb_stat()));
            Ok(stat)
        }
    }

    /// Retrieves info about this environment.
    pub fn info(&self) -> Result<Info> {
        unsafe {
            let mut info = Info(mem::zeroed());
            lmdb_try!(ffi::mdb_env_info(self.env(), &mut info.0));
            Ok(info)
        }
    }

    /// Retrieves the total number of pages on the freelist.
    ///
    /// Along with `Environment::info()`, this can be used to calculate the exact number
    /// of used pages as well as free pages in this environment.
    ///
    /// ```ignore
    /// let env = Environment::new().open("/tmp/test").unwrap();
    /// let info = env.info().unwrap();
    /// let stat = env.stat().unwrap();
    /// let freelist = env.freelist().unwrap();
    /// let last_pgno = info.last_pgno() + 1; // pgno is 0 based.
    /// let total_pgs = info.map_size() / stat.page_size() as usize;
    /// let pgs_in_use = last_pgno - freelist;
    /// let pgs_free = total_pgs - pgs_in_use;
    /// ```
    ///
    /// Note:
    ///
    /// * LMDB stores all the freelists in the designated database 0 in each environment,
    ///   and the freelist count is stored at the beginning of the value as `libc::size_t`
    ///   in the native byte order.
    ///
    /// * It will create a read transaction to traverse the freelist database.
    pub fn freelist(&self) -> Result<size_t> {
        let mut freelist: size_t = 0;
        let db = Database::freelist_db();
        let txn = self.begin_ro_txn()?;
        let cursor = txn.open_ro_cursor(db)?;

        for result in cursor.into_iter() {
            let (_key, value) = result?;
            if value.len() < mem::size_of::<size_t>() {
                return Err(Error::Corrupted);
            }

            let s = &value[..mem::size_of::<size_t>()];
            if cfg!(target_pointer_width = "64") {
                freelist += NativeEndian::read_u64(s) as size_t;
            } else {
                freelist += NativeEndian::read_u32(s) as size_t;
            }
        }

        Ok(freelist)
    }

    /// Sets the size of the memory map to use for the environment.
    ///
    /// This could be used to resize the map when the environment is already open.
    ///
    /// Note:
    ///
    /// * Transactions live in this process are waited out before the map is replaced, so this
    ///   blocks until they finish. Two caller shapes dead-lock against that wait, both of them
    ///   reachable through `do_resize` as well:
    ///   1. The *calling* thread holds a transaction. That transaction can never drain, because
    ///      dropping it is exactly what this call is waiting for.
    ///   2. *Any* thread opens a transaction while it already holds one (`begin_nested_txn`
    ///      included). The second open parks until this resize finishes, while this resize waits
    ///      for the transaction that same thread is still holding.
    ///
    /// * The size should be a multiple of the OS page size. Any attempt to set
    ///   a size smaller than the space already consumed by the environment will
    ///   be silently changed to the current size of the used space.
    ///
    /// * In the multi-process case, once a process resizes the map, other
    ///   processes need to either re-open the environment, or call set_map_size
    ///   with size 0 to update the environment. Otherwise, new transaction creation
    ///   will fail with `Error::MapResized`.
    pub fn set_map_size(&self, size: size_t) -> Result<()> {
        // - take the resize lock so this remap cannot interleave with one driven by do_resize or by
        //   begin_rw_txn_generic; mdb_env_set_mapsize is not safe to run against itself
        // - the expect covers poisoning only, which means another thread panicked mid-resize and left
        //   the map in an unknown state: a broken invariant, not a runtime condition to hand back
        // - returning it instead is not expressible: Error carries an OS errno and has no poison
        //   variant, so propagating would mean inventing an errno the OS never reported
        let _lock = self.db_resize_lock.lock().expect("Database resize mutex lock failed");
        self.set_map_size_locked(size)
    }

    /// The only place the memory map is replaced, so the interlock that makes replacing it safe is
    /// stated once here instead of at each call site, where a caller could omit it.
    /// The caller must hold `db_resize_lock`; `set_map_size` and `do_resize_locked` are the only
    /// callers and both hold it. That is also what keeps the blocker below single-owner: every
    /// blocker in the crate is created from here, hence always under that lock.
    fn set_map_size_locked(&self, size: size_t) -> Result<()> {
        // - mdb_env_set_mapsize unmaps the file and maps it again, normally at a different address,
        //   so any slice a live transaction handed out would be left dangling
        // - LMDB itself only refuses this while a write transaction is open; read transactions are
        //   invisible to it, so excluding them is on us
        // - blocking new transactions and draining the live ones is what closes that window
        let _tx_blocker = ScopedTransactionBlocker::new(self);
        TransactionGuard::wait_for_transactions_to_finish(self);

        unsafe { lmdb_result(ffi::mdb_env_set_mapsize(self.env(), size)) }
    }

    // size_used doesn't include data yet to be committed. This will work only
    // at the beginning of a transaction
    fn map_occupied_size_inner(env_info: &Info, stat: &Stat) -> usize {
        (stat.page_size() as usize).checked_mul(env_info.last_pgno()).unwrap_or_else(|| {
            panic!("lmdb: Occupied size calculation failed: {} * {}", stat.page_size(), env_info.last_pgno())
        })
    }

    /// Check whether a resize is needed under two conditions:
    /// 1. The headroom + currently used size don't fit in map_size()
    /// 2. The used fraction of the database crosses the configured resize trigger fraction
    fn needs_resize(&self, headroom: Option<usize>) -> Result<bool> {
        let stat = self.stat()?;
        let env_info = self.info()?;

        let size_used = Self::map_occupied_size_inner(&env_info, &stat);

        let current_map_size = env_info.map_size();

        let current_fraction_used = size_used as f32 / current_map_size as f32;

        if let Some(given_headroom) = headroom {
            if env_info.map_size() < given_headroom.checked_add(size_used).expect("LMDB size check addition failed") {
                return Ok(true);
            }
        }

        let resize_settings = self.resize_settings.as_ref().unwrap_or(&DEFAULT_RESIZE_SETTINGS);

        Ok(current_fraction_used > resize_settings.resize_trigger_fraction.as_f32())
    }

    /// Do the resizing. This blocks until every transaction live in this process has finished, then
    /// resizes, and returns the new map size. The same two caller shapes that dead-lock
    /// `set_map_size` dead-lock here, for the same reason: a transaction held on the *calling*
    /// thread, or *any* thread opening a transaction while it already holds one.
    /// Keep in mind that a single resize step cannot be larger than 1 << 31, due to usize limitations
    /// this is due to the FFI using usize while lmdb uses mdb_size_t, which is always u64
    pub fn do_resize(&self, increase_size: Option<usize>) -> Result<usize> {
        // - take the resize lock so two remaps can never run concurrently on the same map
        // - begin_rw_txn_generic already holds this lock when it drives a resize through do_resize_locked,
        //   so the locked region always has exactly one owner
        // - the expect covers poisoning only, which means another thread panicked mid-resize and left
        //   the map in an unknown state: a broken invariant, not a runtime condition to hand back
        // - returning it instead is not expressible: Error carries an OS errno and has no poison
        //   variant, so propagating would mean inventing an errno the OS never reported
        let _lock = self.db_resize_lock.lock().expect("Database resize mutex lock failed");
        self.do_resize_locked(increase_size)
    }

    /// The actual resize step, split out so the resize lock is acquired in exactly one place.
    /// The caller must hold `db_resize_lock`; `do_resize` and `resize_db_if_necessary` are the only
    /// callers and both hold it, which is what makes the transaction draining below single-owner.
    fn do_resize_locked(&self, increase_size: Option<usize>) -> Result<usize> {
        let stat = self.stat()?;
        let env_info = self.info()?;
        let system_page_size = stat.page_size() as usize;

        let old_map_size = env_info.map_size();

        let resize_settings = self.resize_settings.as_ref().unwrap_or(&DEFAULT_RESIZE_SETTINGS);
        let increase_size = increase_size.unwrap_or_else(|| {
            Self::headroom_from_ratio(old_map_size, resize_settings.default_resize_ratio_percentage)
        });
        let increase_size = increase_size.clamp(resize_settings.min_resize_step, resize_settings.max_resize_step);

        let current_occupied_ratio = Self::map_occupied_size_inner(&env_info, &stat);

        // calculate new map size, and ensure it's an integer of OS page size
        let new_map_size = old_map_size.checked_add(increase_size).expect("LMDB resize size addition failed");
        let new_map_size = if new_map_size % system_page_size != 0 {
            new_map_size + (system_page_size - new_map_size % system_page_size)
        } else {
            new_map_size
        };

        // To prevent dead-locks (where we keep retrying to resize to the same size), we ensure that we're at least increasing the size by a page's size
        let new_map_size = if new_map_size == old_map_size {
            new_map_size + system_page_size
        } else {
            new_map_size
        };

        // Check the invariants of the resize
        assert_eq!(
            new_map_size % system_page_size,
            0,
            "Attempted resize with size {} not equal to integers of page size {}",
            new_map_size,
            stat.page_size()
        );
        assert!(
            new_map_size > old_map_size,
            "Attempted resize with new size <= old size: {} <= {}",
            new_map_size,
            old_map_size
        );

        // Check available disk space
        // - a failed free-space query is a runtime filesystem error, not a broken invariant, so return it
        // - route it onto the OS-error channel LMDB already uses; EIO covers the rare io error with no OS code
        let free_space =
            fs4::free_space(&self.db_path).map_err(|e| Error::Other(e.raw_os_error().unwrap_or(libc::EIO)))?;
        let final_increase = new_map_size
            .checked_sub(old_map_size)
            .expect("Resize invariant broken: new_map_size < old_map_size") as u64;
        // - refuse to grow past what the disk can back; a full disk is a normal runtime condition, not a panic
        // - ENOSPC mirrors what the OS itself reports when a mapping cannot be backed by disk
        if free_space <= final_increase {
            return Err(Error::Other(libc::ENOSPC));
        }

        // - excluding live transactions is part of replacing the map, so it lives in set_map_size_locked
        // - the locked variant is the one called here because this function already holds db_resize_lock
        self.set_map_size_locked(new_map_size)?;

        if let Some(resize_callback) = &self.resize_callback {
            resize_callback(DatabaseResizeInfo {
                old_size: old_map_size as u64,
                new_size: new_map_size as u64,
                occupied_size_before_resize: current_occupied_ratio as u64,
            });
        }

        Ok(new_map_size)
    }
}

/// Environment statistics.
///
/// Contains information about the size and layout of an LMDB environment or database.
pub struct Stat(ffi::MDB_stat);

impl Stat {
    /// Create a new Stat with zero'd inner struct `ffi::MDB_stat`.
    pub(crate) fn new() -> Stat {
        unsafe { Stat(mem::zeroed()) }
    }

    /// Returns a mut pointer to `ffi::MDB_stat`.
    pub(crate) fn mdb_stat(&mut self) -> *mut ffi::MDB_stat {
        &mut self.0
    }
}

impl Stat {
    /// Size of a database page. This is the same for all databases in the environment.
    #[inline]
    pub fn page_size(&self) -> u32 {
        self.0.ms_psize
    }

    /// Depth (height) of the B-tree.
    #[inline]
    pub fn depth(&self) -> u32 {
        self.0.ms_depth
    }

    /// Number of internal (non-leaf) pages.
    #[inline]
    pub fn branch_pages(&self) -> usize {
        self.0.ms_branch_pages
    }

    /// Number of leaf pages.
    #[inline]
    pub fn leaf_pages(&self) -> usize {
        self.0.ms_leaf_pages
    }

    /// Number of overflow pages.
    #[inline]
    pub fn overflow_pages(&self) -> usize {
        self.0.ms_overflow_pages
    }

    /// Number of data items.
    #[inline]
    pub fn entries(&self) -> usize {
        self.0.ms_entries
    }
}

/// Environment information.
///
/// Contains environment information about the map size, readers, last txn id etc.
pub struct Info(ffi::MDB_envinfo);

impl Info {
    /// Size of memory map.
    #[inline]
    #[must_use]
    pub fn map_size(&self) -> usize {
        self.0.me_mapsize
    }

    /// Last used page number
    #[inline]
    #[must_use]
    pub fn last_pgno(&self) -> usize {
        self.0.me_last_pgno
    }

    /// Last transaction ID
    #[inline]
    #[must_use]
    pub fn last_txnid(&self) -> usize {
        self.0.me_last_txnid
    }

    /// Max reader slots in the environment
    #[inline]
    #[must_use]
    pub fn max_readers(&self) -> u32 {
        self.0.me_maxreaders
    }

    /// Max reader slots used in the environment
    #[inline]
    #[must_use]
    pub fn num_readers(&self) -> u32 {
        self.0.me_numreaders
    }
}

// - the two unsafe impls below are unconditional, so the compiler stops re-deriving Send and Sync
//   from the fields and never complains again no matter what the fields become
// - what makes them sound is an audit: every field except the raw env pointer is safe to move and
//   share across threads on its own, and the pointer itself is safe because LMDB does its own
//   locking behind it
// - the boxed resize callback is the weakest link, because its thread-safety comes from a bound
//   written by hand on the field type; dropping that bound would let a caller hand over a closure
//   that is not safe to share, and nothing would catch it
// - passing each audited field through a gate that demands Send + Sync turns a dropped bound, or a
//   field swapped for a type that is not shareable, into a build error
// - a newly added field still has to be added here by hand; the gate pins the audit as written, it
//   does not discover fields on its own
// - the raw MDB_env pointer is deliberately absent: it is the one field that cannot pass the gate,
//   and it is the whole reason these unsafe impls are written by hand
fn _assert_environment_audited_fields_are_send_sync(env: &Environment) {
    fn gate<T: Send + Sync + ?Sized>(_: &T) {}

    gate(&env.tx_count);
    gate(&env.tx_blocker_count);
    gate(&env.db_resize_lock);
    gate(&env.dbi_open_mutex);
    gate(&env.resize_callback);
    gate(&env.resize_settings);
    gate(&env.db_path);
}

unsafe impl Send for Environment {}
unsafe impl Sync for Environment {}

impl fmt::Debug for Environment {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Environment").finish()
    }
}

impl Drop for Environment {
    fn drop(&mut self) {
        // This is a solution for the issue where, very rarely, closing an environment
        // from a thread where a transaction was executed causes a SIGSEGV.
        // This issue was proven and tested under rare circumstances
        std::thread::scope(|s| {
            s.spawn(move || unsafe { ffi::mdb_env_close(self.env) })
                .join()
                .expect("Failed to join lmdb Drop for Environment thread");
        });
    }
}

///////////////////////////////////////////////////////////////////////////////////////////////////
////// Environment Builder
///////////////////////////////////////////////////////////////////////////////////////////////////

/// Options for opening or creating an environment.
pub struct EnvironmentBuilder {
    flags: EnvironmentFlags,
    max_readers: Option<c_uint>,
    max_dbs: Option<c_uint>,
    map_size: Option<size_t>,
    resize_callback: Option<Box<dyn Fn(DatabaseResizeInfo) + Send + Sync>>,
    resize_settings: Option<DatabaseResizeSettings>,
}

impl EnvironmentBuilder {
    /// Open an environment.
    ///
    /// On UNIX, the database files will be opened with 644 permissions.
    ///
    /// The path may not contain the null character, Windows UNC (Uniform Naming Convention)
    /// paths are not supported either.
    pub fn open(self, path: &Path) -> Result<Environment> {
        self.open_with_permissions(path, 0o644)
    }

    /// Open an environment with the provided UNIX permissions.
    ///
    /// On Windows, the permissions will be ignored.
    ///
    /// The path may not contain the null character, Windows UNC (Uniform Naming Convention)
    /// paths are not supported either.
    pub fn open_with_permissions(self, path: &Path, mode: ffi::mdb_mode_t) -> Result<Environment> {
        let mut env: *mut ffi::MDB_env = ptr::null_mut();
        unsafe {
            lmdb_try!(ffi::mdb_env_create(&mut env));
            if let Some(max_readers) = self.max_readers {
                lmdb_try_with_cleanup!(ffi::mdb_env_set_maxreaders(env, max_readers), ffi::mdb_env_close(env))
            }
            if let Some(max_dbs) = self.max_dbs {
                lmdb_try_with_cleanup!(ffi::mdb_env_set_maxdbs(env, max_dbs), ffi::mdb_env_close(env))
            }
            if let Some(map_size) = self.map_size {
                lmdb_try_with_cleanup!(ffi::mdb_env_set_mapsize(env, map_size), ffi::mdb_env_close(env))
            }
            let path = match CString::new(path.as_os_str().as_bytes()) {
                Ok(path) => path,
                Err(..) => return Err(crate::Error::Invalid),
            };
            lmdb_try_with_cleanup!(
                ffi::mdb_env_open(env, path.as_ptr(), self.flags.bits(), mode),
                ffi::mdb_env_close(env)
            );
        }
        Ok(Environment {
            env,
            tx_count: AtomicU32::new(0),
            tx_blocker_count: AtomicU32::new(0),
            db_resize_lock: Mutex::new(()),
            dbi_open_mutex: Mutex::new(()),
            resize_callback: self.resize_callback,
            resize_settings: self.resize_settings,
            db_path: path.to_owned(),
        })
    }

    /// Sets the provided options in the environment.
    pub fn set_flags(mut self, flags: EnvironmentFlags) -> EnvironmentBuilder {
        self.flags = flags;
        self
    }

    /// Sets the maximum number of threads or reader slots for the environment.
    ///
    /// This defines the number of slots in the lock table that is used to track readers in the
    /// the environment. The default is 126. Starting a read-only transaction normally ties a lock
    /// table slot to the current thread until the environment closes or the thread exits. If
    /// `MDB_NOTLS` is in use, `Environment::open_txn` instead ties the slot to the `Transaction`
    /// object until it or the `Environment` object is destroyed.
    pub fn set_max_readers(mut self, max_readers: c_uint) -> EnvironmentBuilder {
        self.max_readers = Some(max_readers);
        self
    }

    /// Sets the maximum number of named databases for the environment.
    ///
    /// This function is only needed if multiple databases will be used in the
    /// environment. Simpler applications that use the environment as a single
    /// unnamed database can ignore this option.
    ///
    /// Currently a moderate number of slots are cheap but a huge number gets
    /// expensive: 7-120 words per transaction, and every `Transaction::open_db`
    /// does a linear search of the opened slots.
    pub fn set_max_dbs(mut self, max_dbs: c_uint) -> EnvironmentBuilder {
        self.max_dbs = Some(max_dbs);
        self
    }

    /// Sets the size of the memory map to use for the environment.
    ///
    /// The size should be a multiple of the OS page size. The default is
    /// 1048576 bytes. The size of the memory map is also the maximum size
    /// of the database. The value should be chosen as large as possible,
    /// to accommodate future growth of the database. It may be increased at
    /// later times.
    ///
    /// Any attempt to set a size smaller than the space already consumed
    /// by the environment will be silently changed to the current size of the used space.
    pub fn set_map_size(mut self, map_size: size_t) -> EnvironmentBuilder {
        self.map_size = Some(map_size);
        self
    }

    /// Set the function that will be called when a database resize happens.
    ///
    /// The callback runs on whichever thread drove the resize, and that thread still holds the
    /// environment's internal resize lock while the callback runs. That lock is not reentrant, so
    /// calling `set_map_size`, `do_resize`, or `begin_rw_txn` from inside the callback dead-locks the
    /// callback's own thread against itself.
    ///
    /// Opening a read transaction with `begin_ro_txn` is allowed: the map has already been replaced
    /// by the time the callback runs, and the block on new transactions is lifted before the call.
    ///
    /// The callback is stored in the [`Environment`], which can be shared across threads, so the
    /// callback has to be safe to send and share too. A closure that captures something which is
    /// not, such as an [`Rc`](std::rc::Rc), is refused at compile time:
    ///
    /// ```compile_fail,E0277
    /// use lmdb::{DatabaseResizeInfo, Environment};
    /// use std::rc::Rc;
    ///
    /// let not_send = Rc::new(());
    /// Environment::new().set_resize_callback(Some(Box::new(move |_info: DatabaseResizeInfo| {
    ///     drop(not_send.clone());
    /// })));
    /// ```
    ///
    /// A callback that captures only shareable state is accepted:
    ///
    /// ```
    /// use lmdb::{DatabaseResizeInfo, Environment};
    /// use std::sync::atomic::{AtomicU64, Ordering};
    /// use std::sync::Arc;
    ///
    /// let resize_count = Arc::new(AtomicU64::new(0));
    /// let counter = Arc::clone(&resize_count);
    /// Environment::new().set_resize_callback(Some(Box::new(move |_info: DatabaseResizeInfo| {
    ///     counter.fetch_add(1, Ordering::Relaxed);
    /// })));
    /// ```
    pub fn set_resize_callback(
        mut self,
        callback: Option<Box<dyn Fn(DatabaseResizeInfo) + Send + Sync>>,
    ) -> EnvironmentBuilder {
        self.resize_callback = callback;
        self
    }

    /// The settings that control when and how resize happens
    pub fn set_resize_settings(mut self, settings: DatabaseResizeSettings) -> EnvironmentBuilder {
        settings.validate();
        self.resize_settings = Some(settings);
        self
    }
}

#[cfg(test)]
mod test {
    use std::sync::atomic::{AtomicBool, AtomicUsize, Ordering};
    use std::sync::mpsc;
    use std::sync::{OnceLock, Weak};
    use std::thread;
    use std::time::Duration;
    use std::{collections::BTreeMap, sync::Arc};

    use byteorder::{ByteOrder, LittleEndian};
    use rand::rngs::StdRng;
    use rand::{Rng, SeedableRng};
    use tempdir::TempDir;

    use crate::flags::*;
    use crate::resize::ResizeTriggerFraction;

    use super::*;

    #[test]
    fn test_open() {
        let dir = TempDir::new("test").unwrap();

        // opening non-existent env with read-only should fail
        assert!(Environment::new().set_flags(EnvironmentFlags::READ_ONLY).open(dir.path()).is_err());

        // opening non-existent env should succeed
        assert!(Environment::new().open(dir.path()).is_ok());

        // opening env with read-only should succeed
        assert!(Environment::new().set_flags(EnvironmentFlags::READ_ONLY).open(dir.path()).is_ok());
    }

    #[test]
    fn test_begin_txn() {
        let dir = TempDir::new("test").unwrap();

        {
            // writable environment
            let env = Environment::new().open(dir.path()).unwrap();

            assert!(env.begin_rw_txn(None).is_ok());
            assert!(env.begin_ro_txn().is_ok());
        }

        {
            // read-only environment
            let env = Environment::new().set_flags(EnvironmentFlags::READ_ONLY).open(dir.path()).unwrap();

            assert!(env.begin_rw_txn(None).is_err());
            assert!(env.begin_ro_txn().is_ok());
        }
    }

    #[test]
    fn test_open_db() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().set_max_dbs(1).open(dir.path()).unwrap();

        assert!(env.open_db(None).is_ok());
        assert!(env.open_db(Some("testdb")).is_err());
    }

    #[test]
    fn test_create_db() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().set_max_dbs(11).open(dir.path()).unwrap();
        assert!(env.open_db(Some("testdb")).is_err());
        assert!(env.create_db(Some("testdb"), DatabaseFlags::empty()).is_ok());
        assert!(env.open_db(Some("testdb")).is_ok())
    }

    #[test]
    fn test_close_database() {
        let dir = TempDir::new("test").unwrap();
        let mut env = Environment::new().set_max_dbs(10).open(dir.path()).unwrap();

        let db = env.create_db(Some("db"), DatabaseFlags::empty()).unwrap();
        unsafe {
            env.close_db(db);
        }
        assert!(env.open_db(Some("db")).is_ok());
    }

    #[test]
    fn test_sync() {
        let dir = TempDir::new("test").unwrap();
        {
            let env = Environment::new().open(dir.path()).unwrap();
            assert!(env.sync(true).is_ok());
        }
        {
            let env = Environment::new().set_flags(EnvironmentFlags::READ_ONLY).open(dir.path()).unwrap();
            assert!(env.sync(true).is_err());
        }
    }

    #[test]
    fn test_stat() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();

        // Stats should be empty initially.
        let stat = env.stat().unwrap();
        assert_eq!(stat.page_size(), page_size::get() as u32);
        assert_eq!(stat.depth(), 0);
        assert_eq!(stat.branch_pages(), 0);
        assert_eq!(stat.leaf_pages(), 0);
        assert_eq!(stat.overflow_pages(), 0);
        assert_eq!(stat.entries(), 0);

        let db = env.open_db(None).unwrap();

        // Write a few small values.
        for i in 0..64 {
            let mut value = [0u8; 8];
            LittleEndian::write_u64(&mut value, i);
            let mut tx = env.begin_rw_txn(None).expect("begin_rw_txn");
            tx.put(db, &value, &value, WriteFlags::default()).expect("tx.put");
            tx.commit().expect("tx.commit")
        }

        // Stats should now reflect inserted values.
        let stat = env.stat().unwrap();
        assert_eq!(stat.page_size(), page_size::get() as u32);
        assert_eq!(stat.depth(), 1);
        assert_eq!(stat.branch_pages(), 0);
        assert_eq!(stat.leaf_pages(), 1);
        assert_eq!(stat.overflow_pages(), 0);
        assert_eq!(stat.entries(), 64);
    }

    #[test]
    fn test_info() {
        let map_size = 1024 * 1024;
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().set_map_size(map_size).open(dir.path()).unwrap();

        let info = env.info().unwrap();
        assert_eq!(info.map_size(), map_size);
        assert_eq!(info.last_pgno(), 1);
        assert_eq!(info.last_txnid(), 0);
        // The default max readers is 126.
        assert_eq!(info.max_readers(), 126);
        assert_eq!(info.num_readers(), 0);
    }

    #[test]
    fn test_freelist() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();

        let db = env.open_db(None).unwrap();
        let mut freelist = env.freelist().unwrap();
        assert_eq!(freelist, 0);

        // Write a few small values.
        for i in 0..64 {
            let mut value = [0u8; 8];
            LittleEndian::write_u64(&mut value, i);
            let mut tx = env.begin_rw_txn(None).expect("begin_rw_txn");
            tx.put(db, &value, &value, WriteFlags::default()).expect("tx.put");
            tx.commit().expect("tx.commit")
        }
        let mut tx = env.begin_rw_txn(None).expect("begin_rw_txn");
        tx.clear_db(db).expect("clear");
        tx.commit().expect("tx.commit");

        // Freelist should not be empty after clear_db.
        freelist = env.freelist().unwrap();
        assert!(freelist > 0);
    }

    #[test]
    fn test_set_map_size() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();

        let mut info = env.info().unwrap();
        let default_size = info.map_size();

        // Resizing to 0 merely reloads the map size
        env.set_map_size(0).unwrap();
        info = env.info().unwrap();
        assert_eq!(info.map_size(), default_size);

        env.set_map_size(2 * default_size).unwrap();
        info = env.info().unwrap();
        assert_eq!(info.map_size(), 2 * default_size);

        env.set_map_size(4 * default_size).unwrap();
        info = env.info().unwrap();
        assert_eq!(info.map_size(), 4 * default_size);

        // Decreasing is also fine if the space hasn't been consumed.
        env.set_map_size(2 * default_size).unwrap();
        info = env.info().unwrap();
        assert_eq!(info.map_size(), 2 * default_size);
    }

    // - resizing replaces the memory map: the old one is unmapped, a new one mapped at a new address
    // - a read transaction hands out &[u8] that point straight into that map
    // - so a resize running next to a live reader leaves the reader holding dangling pointers
    // - LMDB only refuses a resize while a write transaction is open, so readers are invisible to it
    // - set_map_size used to remap with no interlock, reachable from safe code with a shared &Environment
    // - set_map_size now drains live transactions first, exactly like do_resize
    // - what is asserted is the interlock, not the corruption: the resize must not return while a
    //   reader is still live
    // - against the uninterlocked version the resize returns immediately, or the reader segfaults
    //   reading through the freed mapping
    #[test]
    fn test_set_map_size_waits_for_live_reader() {
        const READER_HOLD: Duration = Duration::from_millis(200);

        let dir = TempDir::new("test").unwrap();
        let env = Arc::new(Environment::new().set_map_size(1 << 20).open(dir.path()).unwrap());
        let db = env.create_db(None, DatabaseFlags::empty()).unwrap();

        let value = vec![0xABu8; 4096];
        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            txn.put(db, b"k", &value, WriteFlags::empty()).unwrap();
            txn.commit().unwrap();
        }

        let reader_live = Arc::new(AtomicBool::new(false));
        let reader_finished = Arc::new(AtomicBool::new(false));

        let reader_env = Arc::clone(&env);
        let reader_live_flag = Arc::clone(&reader_live);
        let reader_finished_flag = Arc::clone(&reader_finished);
        let expected = value.clone();
        let reader = thread::spawn(move || {
            let txn = reader_env.begin_ro_txn().unwrap();
            // borrows straight into the memory map that the resizer is about to replace
            let mapped: &[u8] = txn.get(db, b"k").unwrap();
            reader_live_flag.store(true, Ordering::SeqCst);
            thread::sleep(READER_HOLD);
            // reading through the slice after a racing resize is the use-after-unmap
            assert_eq!(mapped, expected.as_slice(), "reader observed a mapping replaced underneath it");
            // published before the transaction guard is released, so a resizer that waited for this
            // reader to drain is guaranteed to observe it
            reader_finished_flag.store(true, Ordering::SeqCst);
            txn.abort();
        });

        while !reader_live.load(Ordering::SeqCst) {
            thread::yield_now();
        }
        env.set_map_size(4 << 20).unwrap();
        assert!(
            reader_finished.load(Ordering::SeqCst),
            "set_map_size replaced the memory map while a read transaction was still live"
        );

        reader.join().unwrap();
    }

    // - randomized companion to test_set_map_size_waits_for_live_reader
    // - covers the input space that matters: reader count, how long each reader holds its slice,
    //   the stored value, and the target map size
    #[test]
    fn test_set_map_size_waits_for_live_readers_randomized() {
        // Seed is printed so a failing run can be replayed by hardcoding it.
        let seed: u64 = rand::random();
        println!("test_set_map_size_waits_for_live_readers_randomized seed: {seed}");
        // Replaying a hardcoded seed is valid only while rand stays pinned to
        // 0.8; StdRng's algorithm is not stable across rand major versions.
        // A replay reproduces the RNG-derived inputs only, not the thread interleaving, which no
        // seed controls; that is inherent to a concurrency test.
        let mut rng = StdRng::seed_from_u64(seed);

        for round in 0..4u32 {
            let n_readers = rng.gen_range(1..=6usize);
            let hold = Duration::from_millis(rng.gen_range(50..=150));
            let new_map_size = rng.gen_range(2..=8usize) * (1usize << 20);
            let value_len = rng.gen_range(1..=4096usize);
            let value: Vec<u8> = (0..value_len).map(|_| rng.gen::<u8>()).collect();

            let dir = TempDir::new("test").unwrap();
            let env = Arc::new(Environment::new().set_map_size(1 << 20).open(dir.path()).unwrap());
            let db = env.create_db(None, DatabaseFlags::empty()).unwrap();
            {
                let mut txn = env.begin_rw_txn(None).unwrap();
                txn.put(db, b"k", &value, WriteFlags::empty()).unwrap();
                txn.commit().unwrap();
            }

            let readers_live = Arc::new(AtomicUsize::new(0));
            let readers_finished = Arc::new(AtomicUsize::new(0));
            let mut handles = Vec::with_capacity(n_readers);
            for _ in 0..n_readers {
                let reader_env = Arc::clone(&env);
                let live = Arc::clone(&readers_live);
                let finished = Arc::clone(&readers_finished);
                let expected = value.clone();
                handles.push(thread::spawn(move || {
                    let txn = reader_env.begin_ro_txn().unwrap();
                    // borrows straight into the memory map that the resizer is about to replace
                    let mapped: &[u8] = txn.get(db, b"k").unwrap();
                    live.fetch_add(1, Ordering::SeqCst);
                    thread::sleep(hold);
                    assert_eq!(
                        mapped,
                        expected.as_slice(),
                        "seed {seed}, round {round}: reader observed a mapping replaced underneath it"
                    );
                    // published before the transaction guard is released, so a resizer that drained
                    // this reader is guaranteed to observe it
                    finished.fetch_add(1, Ordering::SeqCst);
                    txn.abort();
                }));
            }

            while readers_live.load(Ordering::SeqCst) != n_readers {
                thread::yield_now();
            }
            env.set_map_size(new_map_size).unwrap();
            let still_live = n_readers - readers_finished.load(Ordering::SeqCst);
            assert_eq!(
                still_live, 0,
                "seed {seed}, round {round}: set_map_size replaced the memory map while {still_live} reader(s) were still live"
            );

            for handle in handles {
                handle.join().unwrap();
            }
        }
    }

    #[must_use]
    fn create_random_data_map_with_target_byte_size(
        required_size: usize,
        key_max_size: usize,
        val_max_size: usize,
    ) -> BTreeMap<Vec<u8>, Vec<u8>> {
        let mut result = BTreeMap::new();

        let mut total_size = 0;

        while total_size < required_size {
            let key_size = 1 + rand::random::<usize>() % key_max_size;
            let key = (0..key_size).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
            let val_size = 1 + rand::random::<usize>() % val_max_size;
            let val = (0..val_size).map(|_| rand::random::<u8>()).collect::<Vec<_>>();
            result.insert(key, val);

            total_size += key_size;
            total_size += val_size;
        }

        result
    }

    #[test]
    fn test_auto_resize() {
        let resize_actions = Arc::new(Mutex::new(Vec::new()));

        let resize_actions_for_check = Arc::clone(&resize_actions);
        let resize_actions = Arc::clone(&resize_actions);
        let resize_callback = Box::new(move |v| resize_actions.lock().unwrap().push(v));

        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 20,
            max_resize_step: 1 << 21,
            default_resize_ratio_percentage: 10,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };

        let dir = TempDir::new("test").unwrap();
        let initial_map_size = 1 << 20;
        let env = Environment::new()
            .set_map_size(initial_map_size)
            .set_resize_callback(Some(resize_callback))
            .set_resize_settings(resize_settings.clone())
            .open(dir.path())
            .unwrap();
        let db = env.create_db(None, DatabaseFlags::default()).unwrap();

        let info = env.info().unwrap();
        let map_size = info.map_size();

        assert_eq!(initial_map_size, map_size);

        // generate random values with a predefined target size that surpasses the current map size
        let data = create_random_data_map_with_target_byte_size(initial_map_size * 5, 500, 10000);

        // write many key/val values, and while they're being written, expect that database map will grow
        for (key, val) in &data {
            let mut rw_tx = env.begin_rw_txn(None).unwrap();
            rw_tx.put(db, &key, &val, WriteFlags::empty()).unwrap();
            rw_tx.commit().unwrap();
        }

        // check resize steps
        let resize_action_result = resize_actions_for_check.lock().unwrap().clone();
        assert!(!resize_action_result.is_empty());
        for act in resize_action_result {
            assert!(act.old_size < act.new_size);
            assert!(act.new_size - act.old_size >= resize_settings.min_resize_step as u64);
            assert!(act.new_size - act.old_size <= resize_settings.max_resize_step as u64);
        }

        // ensure data is successfully written
        let ro_tx = env.begin_ro_txn().unwrap();
        for (key, val) in data {
            assert_eq!(ro_tx.get(db, &key).unwrap(), val);
        }
    }

    #[test]
    fn test_slow_auto_resize() {
        let resize_actions = Arc::new(Mutex::new(Vec::new()));

        let resize_actions_for_check = Arc::clone(&resize_actions);
        let resize_actions = Arc::clone(&resize_actions);
        let resize_callback = Box::new(move |v| resize_actions.lock().unwrap().push(v));

        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 19,
            max_resize_step: 1 << 21,
            default_resize_ratio_percentage: 20,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };

        let dir = TempDir::new("test").unwrap();
        let initial_map_size = 1 << 19;
        let env = Environment::new()
            .set_map_size(initial_map_size)
            .set_resize_callback(Some(resize_callback))
            .set_resize_settings(resize_settings.clone())
            .open(dir.path())
            .unwrap();
        let db = env.create_db(None, DatabaseFlags::default()).unwrap();

        let info = env.info().unwrap();
        let map_size = info.map_size();

        assert_eq!(initial_map_size, map_size);

        // generate small random values with a predefined target size that surpasses the current map size
        let data = create_random_data_map_with_target_byte_size(initial_map_size, 5, 10);

        // write many key/val values, and while they're being written, expect that database map will grow
        for (key, val) in &data {
            let mut rw_tx = env.begin_rw_txn(None).unwrap();
            rw_tx.put(db, &key, &val, WriteFlags::empty()).unwrap();
            rw_tx.commit().unwrap();
        }

        // check resize steps
        let resize_action_result = resize_actions_for_check.lock().unwrap().clone();
        assert!(!resize_action_result.is_empty());
        for act in resize_action_result {
            assert!(act.old_size < act.new_size);
            assert!(act.new_size - act.old_size >= resize_settings.min_resize_step as u64);
            assert!(act.new_size - act.old_size <= resize_settings.max_resize_step as u64);
            // make sure that we always crossed the provided threshold before resizing
            assert!(
                act.occupied_size_before_resize as f32 / act.old_size as f32
                    >= resize_settings.resize_trigger_fraction.as_f32(),
                "resize ratio check failed: {} / {} >= {}",
                act.occupied_size_before_resize as f32,
                act.old_size as f32,
                resize_settings.resize_trigger_fraction.as_f32()
            )
        }

        // ensure data is successfully written
        let ro_tx = env.begin_ro_txn().unwrap();
        for (key, val) in data {
            assert_eq!(ro_tx.get(db, &key).unwrap(), val);
        }
    }

    #[test]
    fn test_extremely_slow_resize_and_recover_from_mapfull_error() {
        let resize_actions = Arc::new(Mutex::new(Vec::new()));

        let resize_actions_for_check = Arc::clone(&resize_actions);
        let resize_actions = Arc::clone(&resize_actions);
        let resize_callback = Box::new(move |v| resize_actions.lock().unwrap().push(v));

        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 15,
            max_resize_step: 1 << 21,
            default_resize_ratio_percentage: 1,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };

        let dir = TempDir::new("test").unwrap();
        let initial_map_size = 1 << 12;
        let env = Environment::new()
            .set_map_size(initial_map_size)
            .set_resize_callback(Some(resize_callback))
            .set_resize_settings(resize_settings.clone())
            .open(dir.path())
            .unwrap();
        let db = env.create_db(None, DatabaseFlags::default()).unwrap();

        // generate small random values with a predefined target size that surpasses the current map size
        let data = create_random_data_map_with_target_byte_size(initial_map_size * 256, 2, 5);

        let mut write_resize_count = 0;
        let mut commit_resize_count = 0;

        // write many key/val values, and while they're being written, expect that database map will grow
        for (key, val) in &data {
            loop {
                let mut rw_tx = env.begin_rw_txn(None).unwrap();
                match rw_tx.put(db, &key, &val, WriteFlags::empty()) {
                    Ok(_) => (), // Success in writing value, let's continue to commit
                    Err(e) => match e {
                        Error::MapFull => {
                            println!("Resizing on write...");
                            write_resize_count += 1;
                            drop(rw_tx);
                            env.do_resize(None).unwrap();
                            continue; // resized, let's try again in the inner loop
                        },
                        _ => panic!("Error on put: {}", e),
                    },
                }

                match rw_tx.commit() {
                    Ok(_) => break, // Success in committing value, we can exit the inner loop
                    Err(e) => match e {
                        Error::MapFull => {
                            println!("Resizing on commit...");
                            commit_resize_count += 1;
                            env.do_resize(None).unwrap();
                        },
                        _ => panic!("Error on commit: {}", e),
                    },
                }
            }
        }

        assert!(write_resize_count > 0, "Test failed to trigger write resizes");
        assert!(commit_resize_count > 0, "Test failed to trigger commit resizes");

        // check resize steps
        let resize_action_result = resize_actions_for_check.lock().unwrap().clone();
        assert!(!resize_action_result.is_empty());
        for act in resize_action_result {
            assert!(act.old_size < act.new_size);
            assert!(act.new_size - act.old_size >= resize_settings.min_resize_step as u64);
            assert!(act.new_size - act.old_size <= resize_settings.max_resize_step as u64);
        }

        // ensure data is successfully written
        let ro_tx = env.begin_ro_txn().unwrap();
        for (key, val) in data {
            assert_eq!(ro_tx.get(db, &key).unwrap(), val);
        }
    }

    #[test]
    fn test_forced_resize_on_tx_begin() {
        let resize_actions = Arc::new(Mutex::new(Vec::new()));

        let resize_actions_for_check = Arc::clone(&resize_actions);
        let resize_actions = Arc::clone(&resize_actions);
        let resize_callback = Box::new(move |v| resize_actions.lock().unwrap().push(v));

        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 20,
            max_resize_step: 1 << 21,
            default_resize_ratio_percentage: 50,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };

        let dir = TempDir::new("test").unwrap();
        let initial_map_size = 1 << 20;
        let env = Environment::new()
            .set_map_size(initial_map_size)
            .set_resize_callback(Some(resize_callback))
            .set_resize_settings(resize_settings.clone())
            .open(dir.path())
            .unwrap();

        let info = env.info().unwrap();
        let map_size = info.map_size();

        assert_eq!(initial_map_size, map_size);

        let headroom = 1 << 25;
        let new_target_size = map_size + headroom;

        // force resize by starting a read/write transaction with specified headroom
        let rw_tx = env.begin_rw_txn(Some(headroom)).unwrap();
        rw_tx.abort();

        // ensure resizing went as expected
        let resize_action_result = resize_actions_for_check.lock().unwrap().clone();
        assert!(!resize_action_result.is_empty());
        for act in resize_action_result {
            assert!(act.old_size < act.new_size);
            assert!(act.new_size - act.old_size >= resize_settings.min_resize_step as u64);
            assert!(act.new_size - act.old_size <= resize_settings.max_resize_step as u64);
        }

        // ensure the new map size is larger than the new target size
        let info = env.info().unwrap();
        let new_map_size = info.map_size();

        assert!(new_map_size >= new_target_size, "{} >= {} is false", new_map_size, new_target_size);
    }

    #[test]
    fn test_resize_non_integer_page_size() {
        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 17,
            max_resize_step: 1 << 19,
            default_resize_ratio_percentage: 10,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };

        let dir = TempDir::new("test").unwrap();
        let initial_map_size = 1 << 20;
        let env = Environment::new()
            .set_map_size(initial_map_size)
            .set_resize_settings(resize_settings)
            .open(dir.path())
            .unwrap();

        let info = env.info().unwrap();
        let map_size = info.map_size();

        assert_eq!(initial_map_size, map_size);

        // this should work as the function will round up page sizes
        env.do_resize(Some((1 << 17) + 7)).unwrap();
    }

    fn small_step_resize_settings() -> DatabaseResizeSettings {
        DatabaseResizeSettings {
            min_resize_step: 1 << 16,
            max_resize_step: 1 << 18,
            default_resize_ratio_percentage: 10,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        }
    }

    // The disk-space guard must return an error, not panic. Request a growth larger than any real disk
    // (tens of petabytes) so the free-space check fails deterministically on any machine.
    #[test]
    fn test_resize_returns_err_when_disk_too_small() {
        let resize_settings = DatabaseResizeSettings {
            min_resize_step: 1 << 45,
            max_resize_step: 1 << 55,
            default_resize_ratio_percentage: 10,
            resize_trigger_fraction: ResizeTriggerFraction::new(0.9).unwrap(),
        };
        let dir = TempDir::new("test").unwrap();
        let env =
            Environment::new().set_map_size(1 << 20).set_resize_settings(resize_settings).open(dir.path()).unwrap();

        // no filesystem can back this growth, so the resize returns ENOSPC instead of panicking
        assert_eq!(env.do_resize(Some(1 << 55)), Err(Error::Other(libc::ENOSPC)));
    }

    // A reset()+drop must fully release the transaction so a later resize can drain tx_count to zero.
    // Under the previous count leak the resizer spins forever, so the resize runs on a thread with a bounded wait.
    #[test]
    fn test_resize_after_reset_does_not_hang() {
        let dir = TempDir::new("test").unwrap();
        let env = Arc::new(
            Environment::new()
                .set_map_size(1 << 20)
                .set_resize_settings(small_step_resize_settings())
                .open(dir.path())
                .unwrap(),
        );

        let txn = env.begin_ro_txn().unwrap();
        let inactive = txn.reset();
        drop(inactive);
        assert_eq!(env.tx_count().load(Ordering::Relaxed), 0);

        let (sender, receiver) = mpsc::channel();
        let resize_env = Arc::clone(&env);
        let handle = thread::spawn(move || {
            let result = resize_env.do_resize(Some(1 << 16));
            sender.send(result).unwrap();
        });

        match receiver.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(_)) => {},
            Ok(Err(e)) => panic!("resize after reset returned an error: {}", e),
            Err(_) => panic!("resize after reset hung: the transaction interlock leaked a count"),
        }
        handle.join().unwrap();
    }

    // - the resize callback runs once the map has been replaced and once the block on new transactions
    //   has been lifted, so a callback is free to open a read transaction
    // - while that block covered the whole resize, this dead-locked: the callback's own thread waited
    //   for a block that only that same thread could lift
    // - so this pins the block to its narrow scope; widening it back over the callback fails here
    // - the resize runs on its own thread with a bounded wait, so such a regression fails this test
    //   instead of hanging the suite forever
    #[test]
    fn test_resize_callback_can_open_ro_txn() {
        let dir = TempDir::new("test").unwrap();

        // - the callback is owned by the Environment, so it cannot own one back
        // - a Weak published after construction breaks that cycle and still reaches the environment
        let env_slot: Arc<OnceLock<Weak<Environment>>> = Arc::new(OnceLock::new());
        let callback_slot = Arc::clone(&env_slot);
        let callback_opened_txn = Arc::new(AtomicBool::new(false));
        let callback_flag = Arc::clone(&callback_opened_txn);
        let callback = Box::new(move |_info: DatabaseResizeInfo| {
            let env = callback_slot.get().unwrap().upgrade().unwrap();
            // waits forever if the resize still blocks new transactions this late
            let txn = env.begin_ro_txn().unwrap();
            txn.abort();
            callback_flag.store(true, Ordering::SeqCst);
        });

        let env = Arc::new(
            Environment::new()
                .set_map_size(1 << 20)
                .set_resize_settings(small_step_resize_settings())
                .set_resize_callback(Some(callback))
                .open(dir.path())
                .unwrap(),
        );
        // published before any resize can run, so the callback always finds it
        env_slot.set(Arc::downgrade(&env)).unwrap();

        let (sender, receiver) = mpsc::channel();
        let resize_env = Arc::clone(&env);
        let handle = thread::spawn(move || {
            sender.send(resize_env.do_resize(Some(1 << 16))).unwrap();
        });

        match receiver.recv_timeout(Duration::from_secs(30)) {
            Ok(Ok(_)) => {},
            Ok(Err(e)) => panic!("resize with a callback returned an error: {}", e),
            Err(_) => panic!("resize callback hung: it could not open a read transaction"),
        }
        handle.join().unwrap();
        assert!(callback_opened_txn.load(Ordering::SeqCst), "the resize callback never ran");
    }

    // Many reader threads open/reset/renew/close read transactions in a loop while the main thread performs
    // many resizes. Exercises the lock-free interlock under contention; a gross regression shows up as a
    // panic, a hang, or corrupted reads. Non-deterministic, but a strong smoke test for the handshake.
    #[test]
    fn test_concurrent_txn_and_resize_stress() {
        let dir = TempDir::new("test").unwrap();
        let env = Arc::new(
            Environment::new()
                .set_map_size(1 << 20)
                .set_resize_settings(small_step_resize_settings())
                .open(dir.path())
                .unwrap(),
        );
        let db = env.create_db(None, DatabaseFlags::empty()).unwrap();

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            txn.put(db, b"k", b"v", WriteFlags::empty()).unwrap();
            txn.commit().unwrap();
        }

        let n_readers = 8usize;
        let stop = Arc::new(AtomicBool::new(false));
        let mut handles = Vec::with_capacity(n_readers);
        for _ in 0..n_readers {
            let reader_env = Arc::clone(&env);
            let reader_stop = Arc::clone(&stop);
            handles.push(thread::spawn(move || {
                while !reader_stop.load(Ordering::Relaxed) {
                    let txn = reader_env.begin_ro_txn().unwrap();
                    assert_eq!(b"v", txn.get(db, b"k").unwrap());
                    // exercise the reset/renew guard-carry paths under the resize storm too
                    let inactive = txn.reset();
                    let txn = inactive.renew().unwrap();
                    assert_eq!(b"v", txn.get(db, b"k").unwrap());
                    txn.abort();
                }
            }));
        }

        // - alternate the two public resize entry points, which take the resize lock by different routes
        // - do_resize takes it and computes the new size; set_map_size takes it and remaps to a size we pick
        for i in 0..100 {
            if i % 2 == 0 {
                env.do_resize(Some(1 << 16)).unwrap();
            } else {
                let current_map_size = env.info().unwrap().map_size();
                env.set_map_size(current_map_size + (1 << 16)).unwrap();
            }
        }
        stop.store(true, Ordering::Relaxed);
        for handle in handles {
            handle.join().unwrap();
        }

        // data integrity survives the storm, and every transaction has been accounted for
        let txn = env.begin_ro_txn().unwrap();
        assert_eq!(b"v", txn.get(db, b"k").unwrap());
        txn.abort();
        assert_eq!(env.tx_count().load(Ordering::Relaxed), 0);
    }
}
