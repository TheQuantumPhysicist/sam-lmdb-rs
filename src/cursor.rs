use std::marker::PhantomData;
use std::{fmt, mem, ptr, result, slice};

use libc::{EINVAL, c_uint, c_void, size_t};

use crate::database::Database;
use crate::error::{Error, Result, lmdb_result};
use crate::flags::WriteFlags;
use crate::transaction::Transaction;
use lmdb_sys as ffi;

/// An LMDB cursor.
pub trait Cursor<'txn>: Sized {
    /// Returns a raw pointer to the underlying LMDB cursor.
    ///
    /// The caller **must** ensure that the pointer is not used after the
    /// lifetime of the cursor.
    fn cursor(&self) -> *mut ffi::MDB_cursor;

    /// Retrieves a key/data pair from the cursor. Depending on the cursor op,
    /// the current key may be returned.
    ///
    /// The LMDB C header states the lifetime contract for every value the
    /// database hands back, on `MDB_val`:
    ///
    /// > Values returned from the database are valid only until a subsequent
    /// > update operation, or the end of the transaction. Do not modify or
    /// > free them, they commonly point into the database itself.
    ///
    /// - The returned bytes are not a copy. They point straight into LMDB's
    ///   memory map.
    /// - That contract has two expiry conditions, whichever comes first: a
    ///   subsequent update operation, or the end of the transaction.
    /// - Borrowing `&self` (the cursor) is what encodes the first condition.
    ///   `put` and `del` take `&mut self`, so the borrow checker refuses to let
    ///   a returned slice survive a mutation made through the same cursor.
    /// - Borrowing the transaction instead would encode only the second
    ///   condition. This signature avoids that on purpose: it would promise more
    ///   than LMDB delivers, and a `del` would leave a live slice pointing at
    ///   bytes that have already been moved.
    /// - Moving a cursor is not an update operation, so slices from earlier
    ///   positioning calls stay valid across later ones and may be held
    ///   together. That is why positioning takes a shared borrow.
    ///
    /// ```compile_fail,E0502
    /// use lmdb::{Cursor, Environment, Transaction, WriteFlags};
    /// use lmdb_sys::MDB_GET_CURRENT;
    ///
    /// let dir = tempdir::TempDir::new("doctest").unwrap();
    /// let env = Environment::new().open(dir.path()).unwrap();
    /// let db = env.open_db(None).unwrap();
    /// let mut txn = env.begin_rw_txn(None).unwrap();
    /// let mut cursor = txn.open_rw_cursor(db).unwrap();
    /// cursor.put(b"key", b"val", WriteFlags::empty()).unwrap();
    ///
    /// let (_key, value) = cursor.get(None, None, MDB_GET_CURRENT).unwrap();
    /// // Mutating the same cursor may free or overwrite the page `value` points at.
    /// cursor.put(b"key2", b"val2", WriteFlags::empty()).unwrap();
    /// // Using `value` after the mutation is rejected: it borrows the cursor.
    /// println!("{:?}", value);
    /// ```
    fn get<'a>(&'a self, key: Option<&[u8]>, data: Option<&[u8]>, op: c_uint) -> Result<(Option<&'a [u8]>, &'a [u8])> {
        unsafe {
            let mut key_val = slice_to_val(key);
            let mut data_val = slice_to_val(data);
            let key_ptr = key_val.mv_data;
            lmdb_result(ffi::mdb_cursor_get(self.cursor(), &mut key_val, &mut data_val, op))?;
            let key_out = if key_ptr != key_val.mv_data {
                Some(val_to_slice(key_val))
            } else {
                None
            };
            let data_out = val_to_slice(data_val);
            Ok((key_out, data_out))
        }
    }

    /// Iterate over database items. The iterator will begin with item next
    /// after the cursor, and continue until the end of the database. For new
    /// cursors, the iterator will begin with the first item in the database.
    ///
    /// For databases with duplicate data items (`DatabaseFlags::DUP_SORT`), the
    /// duplicate data items of each key will be returned before moving on to
    /// the next key.
    fn into_iter(self) -> Iter<'txn, Self> {
        Iter::new(self, ffi::MDB_NEXT, ffi::MDB_NEXT)
    }

    /// Iterate over database items starting from the beginning of the database.
    ///
    /// For databases with duplicate data items (`DatabaseFlags::DUP_SORT`), the
    /// duplicate data items of each key will be returned before moving on to
    /// the next key.
    fn into_iter_start(self) -> Iter<'txn, Self> {
        Iter::new(self, ffi::MDB_FIRST, ffi::MDB_NEXT)
    }

    /// Iterate over database items starting from the given key.
    ///
    /// For databases with duplicate data items (`DatabaseFlags::DUP_SORT`), the
    /// duplicate data items of each key will be returned before moving on to
    /// the next key.
    fn into_iter_from<K>(self, key: K) -> Iter<'txn, Self>
    where
        K: AsRef<[u8]>,
    {
        match self.get(Some(key.as_ref()), None, ffi::MDB_SET_RANGE) {
            Ok(_) | Err(Error::NotFound) => (),
            Err(error) => return Iter::err(error),
        };
        Iter::new(self, ffi::MDB_GET_CURRENT, ffi::MDB_NEXT)
    }

    /// Iterate over database items in reverse, starting from the end of the
    /// database.
    ///
    /// - Keys are returned in descending order.
    /// - Mirror of `into_iter_start`, with the direction reversed.
    /// - For databases with duplicate data items (`DatabaseFlags::DUP_SORT`),
    ///   the duplicate data items of each key are returned in reverse order
    ///   before moving on to the previous key.
    fn into_iter_rev(self) -> Iter<'txn, Self> {
        Iter::new(self, ffi::MDB_LAST, ffi::MDB_PREV)
    }

    /// Iterate over database items in reverse, starting from the given key.
    ///
    /// - Returns items with keys less than or equal to the given key, in
    ///   descending key order. The given key, when present, is included.
    /// - Mirror of `into_iter_from`, with the direction reversed.
    /// - For databases with duplicate data items (`DatabaseFlags::DUP_SORT`),
    ///   every duplicate of the matched key is returned in reverse order
    ///   before moving on to earlier keys, mirroring `into_iter_from`.
    fn into_iter_from_rev<K>(self, key: K) -> Iter<'txn, Self>
    where
        K: AsRef<[u8]>,
    {
        // - LMDB's forward range-seek (MDB_SET_RANGE) does not report whether it
        //   landed on the requested key or a greater one, and the reverse walk
        //   needs that distinction.
        // - MDB_SET reports an exact match directly as Ok versus NotFound, so it
        //   drives the exact case; MDB_SET_RANGE then splits a greater key from
        //   all-smaller for the remaining keys.
        match self.get(Some(key.as_ref()), None, ffi::MDB_SET) {
            Ok(_) => {
                // - On DUP_SORT data the cursor sits on the first duplicate of
                //   the key; move to the last so the descending walk yields every
                //   duplicate before crossing to earlier keys.
                // - MDB_LAST_DUP fails on a database opened without duplicate
                //   support and leaves the cursor on the key, which is the
                //   position wanted anyway.
                self.get(None, None, ffi::MDB_LAST_DUP).ok();
                Iter::new(self, ffi::MDB_GET_CURRENT, ffi::MDB_PREV)
            },
            Err(Error::NotFound) => match self.get(Some(key.as_ref()), None, ffi::MDB_SET_RANGE) {
                // Landed on a greater key: step back to the nearest smaller key.
                Ok(_) => Iter::new(self, ffi::MDB_PREV, ffi::MDB_PREV),
                // Every key is smaller than the request: start from the last key.
                Err(Error::NotFound) => Iter::new(self, ffi::MDB_LAST, ffi::MDB_PREV),
                Err(error) => Iter::err(error),
            },
            Err(error) => Iter::err(error),
        }
    }

    /// Iterate over the duplicates of the item in the database with the given key.
    fn into_iter_dup_of<K>(self, key: K) -> Iter<'txn, Self>
    where
        K: AsRef<[u8]>,
    {
        match self.get(Some(key.as_ref()), None, ffi::MDB_SET) {
            Ok(_) => (),
            Err(Error::NotFound) => {
                self.get(None, None, ffi::MDB_LAST).ok();
                return Iter::new(self, ffi::MDB_NEXT, ffi::MDB_NEXT);
            },
            Err(error) => return Iter::err(error),
        };
        Iter::new(self, ffi::MDB_GET_CURRENT, ffi::MDB_NEXT_DUP)
    }
}

/// A read-only cursor for navigating the items within a database.
pub struct RoCursor<'txn> {
    cursor: *mut ffi::MDB_cursor,
    _marker: PhantomData<fn() -> &'txn ()>,
}

impl<'txn> Cursor<'txn> for RoCursor<'txn> {
    fn cursor(&self) -> *mut ffi::MDB_cursor {
        self.cursor
    }
}

impl<'txn> fmt::Debug for RoCursor<'txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("RoCursor").finish()
    }
}

impl<'txn> Drop for RoCursor<'txn> {
    fn drop(&mut self) {
        unsafe { ffi::mdb_cursor_close(self.cursor) }
    }
}

impl<'txn> RoCursor<'txn> {
    /// Creates a new read-only cursor in the given database and transaction.
    /// Prefer using `Transaction::open_cursor`.
    pub(crate) fn new<T>(txn: &'txn T, db: Database) -> Result<RoCursor<'txn>>
    where
        T: Transaction,
    {
        let mut cursor: *mut ffi::MDB_cursor = ptr::null_mut();
        unsafe {
            lmdb_result(ffi::mdb_cursor_open(txn.txn(), db.dbi(), &mut cursor))?;
        }
        Ok(RoCursor {
            cursor,
            _marker: PhantomData,
        })
    }
}

/// A read-write cursor for navigating items within a database.
pub struct RwCursor<'txn> {
    cursor: *mut ffi::MDB_cursor,
    _marker: PhantomData<fn() -> &'txn ()>,
}

impl<'txn> Cursor<'txn> for RwCursor<'txn> {
    fn cursor(&self) -> *mut ffi::MDB_cursor {
        self.cursor
    }
}

impl<'txn> fmt::Debug for RwCursor<'txn> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("RwCursor").finish()
    }
}

impl<'txn> Drop for RwCursor<'txn> {
    fn drop(&mut self) {
        unsafe { ffi::mdb_cursor_close(self.cursor) }
    }
}

impl<'txn> RwCursor<'txn> {
    /// Creates a new read-only cursor in the given database and transaction.
    /// Prefer using `RwTransaction::open_rw_cursor`.
    pub(crate) fn new<T>(txn: &'txn T, db: Database) -> Result<RwCursor<'txn>>
    where
        T: Transaction,
    {
        let mut cursor: *mut ffi::MDB_cursor = ptr::null_mut();
        unsafe {
            lmdb_result(ffi::mdb_cursor_open(txn.txn(), db.dbi(), &mut cursor))?;
        }
        Ok(RwCursor {
            cursor,
            _marker: PhantomData,
        })
    }

    /// Puts a key/data pair into the database. The cursor will be positioned at
    /// the new data item, or on failure usually near it.
    ///
    /// ### Position after a put, when walking
    ///
    /// - A put moves the cursor onto the item it just wrote, wherever that key
    ///   sorts. A walk in progress does not resume where it left off, it resumes
    ///   from the new item. Inserting a key that sorts after the current
    ///   position therefore skips everything in between, silently.
    /// - `WriteFlags::CURRENT` overwrites the item the cursor already sits on
    ///   and leaves the position alone, so it is the flag to use for changing
    ///   values during a walk.
    /// - To insert unrelated keys while walking, finish the walk first and
    ///   insert afterward. There is no flag that keeps a walk intact across an
    ///   insert somewhere else.
    /// - This is the opposite of `del`, which LMDB compensates for so a walk
    ///   continues correctly. See the position contract on [`RwCursor::del`].
    pub fn put<K, D>(&mut self, key: &K, data: &D, flags: WriteFlags) -> Result<()>
    where
        K: AsRef<[u8]>,
        D: AsRef<[u8]>,
    {
        let key = key.as_ref();
        let data = data.as_ref();
        let mut key_val: ffi::MDB_val = ffi::MDB_val {
            mv_size: key.len() as size_t,
            mv_data: key.as_ptr() as *mut c_void,
        };
        let mut data_val: ffi::MDB_val = ffi::MDB_val {
            mv_size: data.len() as size_t,
            mv_data: data.as_ptr() as *mut c_void,
        };
        unsafe { lmdb_result(ffi::mdb_cursor_put(self.cursor(), &mut key_val, &mut data_val, flags.bits())) }
    }

    /// Deletes the current key/data pair.
    ///
    /// ### Flags
    ///
    /// `WriteFlags::NO_DUP_DATA` may be used to delete all data items for the
    /// current key, if the database was opened with `DatabaseFlags::DUP_SORT`.
    ///
    /// ### Position after a delete
    ///
    /// - A delete leaves the cursor on the item that followed the deleted one.
    ///   A following `next` returns that item instead of skipping it, and a
    ///   following `prev` needs no compensation. Do not hand-roll a corrective
    ///   step around a delete; it would skip or repeat an item.
    /// - The reason: removing an item shifts the rest of the page down into its
    ///   slot, so the cursor's stored index already denotes the next item. LMDB
    ///   records that a delete happened and has the next step consume the shift
    ///   rather than advance again (mdb.c, `mdb_cursor_next`, the `C_DEL`
    ///   branch near line 6849).
    /// - This matches erasing through an iterator in C++, where the erase call
    ///   returns an iterator to the following element. LMDB keeps the same idea
    ///   as internal state instead of returning it, which makes the behavior
    ///   invisible from this API and easy to compensate for twice.
    ///
    /// ```compile_fail,E0502
    /// use lmdb::{Environment, Transaction, WriteFlags};
    ///
    /// let dir = tempdir::TempDir::new("doctest").unwrap();
    /// let env = Environment::new().open(dir.path()).unwrap();
    /// let db = env.open_db(None).unwrap();
    /// let mut txn = env.begin_rw_txn(None).unwrap();
    /// txn.put(db, b"key", b"val", WriteFlags::empty()).unwrap();
    /// let mut cursor = txn.open_rw_cursor(db).unwrap();
    ///
    /// let (_key, value) = cursor.next().unwrap().unwrap();
    /// // The delete closes the hole by moving the page's remaining bytes over
    /// // the ones `value` points at.
    /// cursor.del(WriteFlags::empty()).unwrap();
    /// // Using `value` after the delete is rejected: it borrows the cursor.
    /// println!("{:?}", value);
    /// ```
    pub fn del(&mut self, flags: WriteFlags) -> Result<()> {
        unsafe { lmdb_result(ffi::mdb_cursor_del(self.cursor(), flags.bits())) }
    }

    /// Positions the cursor on the first item of the database and returns it.
    ///
    /// - Returns `Ok(None)` when the database holds no items.
    /// - The returned slices borrow the cursor and expire on the next `put` or
    ///   `del` through it. See [`Cursor::get`] for the LMDB contract behind
    ///   that.
    pub fn first(&self) -> Result<Option<(&[u8], &[u8])>> {
        self.walk(ffi::MDB_FIRST)
    }

    /// Positions the cursor on the last item of the database and returns it.
    ///
    /// - Returns `Ok(None)` when the database holds no items.
    /// - The returned slices borrow the cursor, as described on
    ///   [`Cursor::get`].
    pub fn last(&self) -> Result<Option<(&[u8], &[u8])>> {
        self.walk(ffi::MDB_LAST)
    }

    /// Advances the cursor to the next item and returns it.
    ///
    /// - Starts at the first item when the cursor has not been positioned yet.
    /// - Returns `Ok(None)` at the end of the database.
    /// - After a `del` this returns the item that followed the deleted one. See
    ///   the position contract on [`RwCursor::del`].
    /// - The returned slices borrow the cursor, as described on
    ///   [`Cursor::get`].
    pub fn next(&self) -> Result<Option<(&[u8], &[u8])>> {
        self.walk(ffi::MDB_NEXT)
    }

    /// Moves the cursor to the previous item and returns it.
    ///
    /// - Starts at the last item when the cursor has not been positioned yet.
    /// - Returns `Ok(None)` at the start of the database.
    /// - The returned slices borrow the cursor, as described on
    ///   [`Cursor::get`].
    pub fn prev(&self) -> Result<Option<(&[u8], &[u8])>> {
        self.walk(ffi::MDB_PREV)
    }

    // - Shared borrow, not `&mut`: a move is not an update operation under the
    //   LMDB contract quoted on `Cursor::get`, so several returned items may be
    //   held at once. Only `put` and `del` take `&mut self` and expire them.
    fn walk(&self, op: c_uint) -> Result<Option<(&[u8], &[u8])>> {
        match self.get(None, None, op) {
            Ok((key, data)) => {
                // - Given no input key, LMDB writes the landing key into the out
                //   value for every positioning op used here, so success always
                //   carries one.
                // - No fallible alternative fits: an absent key would mean LMDB
                //   broke its own contract, not a condition a caller can act on.
                let key = key.expect("LMDB reports the landing key when given no input key");
                Ok(Some((key, data)))
            },
            // A walk that runs past either end is an ordinary end, not a failure.
            Err(Error::NotFound) => Ok(None),
            Err(error) => Err(error),
        }
    }
}

unsafe fn slice_to_val(slice: Option<&[u8]>) -> ffi::MDB_val {
    match slice {
        Some(slice) => ffi::MDB_val {
            mv_size: slice.len() as size_t,
            mv_data: slice.as_ptr() as *mut c_void,
        },
        None => ffi::MDB_val {
            mv_size: 0,
            mv_data: ptr::null_mut(),
        },
    }
}

unsafe fn val_to_slice<'a>(val: ffi::MDB_val) -> &'a [u8] {
    unsafe { slice::from_raw_parts(val.mv_data as *const u8, val.mv_size) }
}

/// An iterator over the key/value pairs in an LMDB database.
///
/// - The iterator owns its cursor and keeps it unreachable on purpose. The items
///   it yields borrow the transaction, not the cursor, so a `put` or `del`
///   through that cursor would expire them while the borrow checker still
///   accepts them.
/// - `into_cursor` hands the cursor back for read-only cursors only, which have
///   no `put` or `del`. Reaching the cursor by any other route would defeat that
///   restriction, so the state below is private rather than a public enum: a
///   public enum has no per-field privacy, and its fields could simply be
///   destructured out.
pub struct Iter<'txn, C> {
    inner: IterInner<'txn, C>,
}

enum IterInner<'txn, C> {
    /// An iterator that returns an error on every call to Iter.next().
    /// Cursor.iter*() creates an Iter of this type when LMDB returns an error
    /// on retrieval of a cursor.  Using this variant instead of returning
    /// an error makes Cursor.iter()* methods infallible, so consumers only
    /// need to check the result of Iter.next().
    Err(Error),

    /// An iterator that returns an Item on calls to Iter.next().
    /// The Item is a Result<(&'txn [u8], &'txn [u8])>, so this variant
    /// might still return an error, if retrieval of the key/value pair
    /// fails for some reason.
    Ok {
        /// The LMDB cursor with which to iterate.
        cursor: C,

        /// The first operation to perform when the consumer calls Iter.next().
        op: c_uint,

        /// The next and subsequent operations to perform.
        next_op: c_uint,

        /// A marker to ensure the iterator doesn't outlive the transaction.
        _marker: PhantomData<fn(&'txn ())>,
    },
}

impl<'txn, C: Cursor<'txn>> Iter<'txn, C> {
    /// Creates a new iterator backed by the given cursor.
    fn new<'t>(cursor: C, op: c_uint, next_op: c_uint) -> Iter<'t, C> {
        Iter {
            inner: IterInner::Ok {
                cursor,
                op,
                next_op,
                _marker: PhantomData,
            },
        }
    }

    /// Creates an iterator that reports the given error on the first `next`.
    fn err<'t>(error: Error) -> Iter<'t, C> {
        Iter {
            inner: IterInner::Err(error),
        }
    }
}

impl<'txn> Iter<'txn, RoCursor<'txn>> {
    /// Reclaims the cursor, left where the iteration stopped.
    ///
    /// - Offered for read-only cursors only, on purpose. This is the sole way
    ///   to get a cursor back out of a live iterator, and the iterator's items
    ///   borrow the transaction rather than the cursor. Handing back a
    ///   read-write cursor would allow a `put` or `del` that expires those items
    ///   while the borrow checker still accepts them.
    /// - A read-only cursor has no `put` or `del`, so reclaiming one cannot
    ///   invalidate anything.
    ///
    /// ```compile_fail,E0599
    /// use lmdb::{Cursor, Environment, Transaction, WriteFlags};
    ///
    /// let dir = tempdir::TempDir::new("doctest").unwrap();
    /// let env = Environment::new().open(dir.path()).unwrap();
    /// let db = env.open_db(None).unwrap();
    /// let mut txn = env.begin_rw_txn(None).unwrap();
    /// txn.put(db, b"key", b"val", WriteFlags::empty()).unwrap();
    ///
    /// let cursor = txn.open_rw_cursor(db).unwrap();
    /// let iter = cursor.into_iter();
    /// // Rejected: a read-write cursor cannot be reclaimed mid-iteration.
    /// let cursor = iter.into_cursor().unwrap();
    /// ```
    ///
    /// Reaching past this method to lift the cursor out by hand is rejected too,
    /// which is what keeps the restriction above meaningful:
    ///
    /// ```compile_fail,E0223
    /// use lmdb::{Cursor, Environment, Transaction, WriteFlags};
    ///
    /// let dir = tempdir::TempDir::new("doctest").unwrap();
    /// let env = Environment::new().open(dir.path()).unwrap();
    /// let db = env.open_db(None).unwrap();
    /// let mut txn = env.begin_rw_txn(None).unwrap();
    /// txn.put(db, b"key", b"val", WriteFlags::empty()).unwrap();
    ///
    /// let cursor = txn.open_rw_cursor(db).unwrap();
    /// let iter = cursor.into_iter_start();
    /// // Rejected: the iterator's state is private, so the cursor cannot be
    /// // destructured out to sidestep the read-only restriction.
    /// match iter {
    ///     lmdb::Iter::Ok { cursor, .. } => { let _ = cursor; },
    ///     _ => {},
    /// }
    /// ```
    pub fn into_cursor(self) -> Result<RoCursor<'txn>> {
        match self.inner {
            IterInner::Err(err) => Err(err),
            IterInner::Ok {
                cursor,
                op: _,
                next_op: _,
                _marker,
            } => Ok(cursor),
        }
    }
}

impl<'txn, C> fmt::Debug for Iter<'txn, C> {
    fn fmt(&self, f: &mut fmt::Formatter) -> result::Result<(), fmt::Error> {
        f.debug_struct("Iter").finish()
    }
}

impl<'txn, C: Cursor<'txn>> Iterator for Iter<'txn, C> {
    type Item = Result<(&'txn [u8], &'txn [u8])>;

    fn next(&mut self) -> Option<Result<(&'txn [u8], &'txn [u8])>> {
        match self.inner {
            IterInner::Ok {
                ref cursor,
                ref mut op,
                next_op,
                _marker,
            } => {
                let mut key = ffi::MDB_val {
                    mv_size: 0,
                    mv_data: ptr::null_mut(),
                };
                let mut data = ffi::MDB_val {
                    mv_size: 0,
                    mv_data: ptr::null_mut(),
                };
                let op = mem::replace(op, next_op);
                unsafe {
                    match ffi::mdb_cursor_get(cursor.cursor(), &mut key, &mut data, op) {
                        ffi::MDB_SUCCESS => Some(Ok((val_to_slice(key), val_to_slice(data)))),
                        // EINVAL can occur when the cursor was previously seeked to a non-existent value,
                        // e.g. iter_from with a key greater than all values in the database.
                        ffi::MDB_NOTFOUND | EINVAL => None,
                        error => Some(Err(Error::from_err_code(error))),
                    }
                }
            },
            IterInner::Err(err) => Some(Err(err)),
        }
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::environment::*;
    use crate::flags::*;
    use ffi::*;
    use rand::rngs::StdRng;
    use rand::{RngExt, SeedableRng};
    use std::collections::{BTreeMap, BTreeSet};
    use tempdir::TempDir;

    #[test]
    fn test_get() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key1", b"val1", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val2", WriteFlags::empty()).unwrap();
        txn.put(db, b"key3", b"val3", WriteFlags::empty()).unwrap();

        let cursor = txn.open_ro_cursor(db).unwrap();
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_FIRST).unwrap());
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_GET_CURRENT).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val2"[..]), cursor.get(None, None, MDB_NEXT).unwrap());
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_PREV).unwrap());
        assert_eq!((Some(&b"key3"[..]), &b"val3"[..]), cursor.get(None, None, MDB_LAST).unwrap());
        assert_eq!((None, &b"val2"[..]), cursor.get(Some(b"key2"), None, MDB_SET).unwrap());
        assert_eq!((Some(&b"key3"[..]), &b"val3"[..]), cursor.get(Some(&b"key3"[..]), None, MDB_SET_KEY).unwrap());
        assert_eq!((Some(&b"key3"[..]), &b"val3"[..]), cursor.get(Some(&b"key2\0"[..]), None, MDB_SET_RANGE).unwrap());
    }

    #[test]
    fn test_get_dup() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key1", b"val1", WriteFlags::empty()).unwrap();
        txn.put(db, b"key1", b"val2", WriteFlags::empty()).unwrap();
        txn.put(db, b"key1", b"val3", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val1", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val2", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val3", WriteFlags::empty()).unwrap();

        let cursor = txn.open_ro_cursor(db).unwrap();
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_FIRST).unwrap());
        assert_eq!((None, &b"val1"[..]), cursor.get(None, None, MDB_FIRST_DUP).unwrap());
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_GET_CURRENT).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val1"[..]), cursor.get(None, None, MDB_NEXT_NODUP).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val2"[..]), cursor.get(None, None, MDB_NEXT_DUP).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val3"[..]), cursor.get(None, None, MDB_NEXT_DUP).unwrap());
        assert!(cursor.get(None, None, MDB_NEXT_DUP).is_err());
        assert_eq!((Some(&b"key2"[..]), &b"val2"[..]), cursor.get(None, None, MDB_PREV_DUP).unwrap());
        assert_eq!((None, &b"val3"[..]), cursor.get(None, None, MDB_LAST_DUP).unwrap());
        assert_eq!((Some(&b"key1"[..]), &b"val3"[..]), cursor.get(None, None, MDB_PREV_NODUP).unwrap());
        assert_eq!((None, &b"val1"[..]), cursor.get(Some(&b"key1"[..]), None, MDB_SET).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val1"[..]), cursor.get(Some(&b"key2"[..]), None, MDB_SET_KEY).unwrap());
        assert_eq!((Some(&b"key2"[..]), &b"val1"[..]), cursor.get(Some(&b"key1\0"[..]), None, MDB_SET_RANGE).unwrap());
        assert_eq!((None, &b"val3"[..]), cursor.get(Some(&b"key1"[..]), Some(&b"val3"[..]), MDB_GET_BOTH).unwrap());
        assert_eq!(
            (None, &b"val1"[..]),
            cursor.get(Some(&b"key2"[..]), Some(&b"val"[..]), MDB_GET_BOTH_RANGE).unwrap()
        );
    }

    #[test]
    fn test_get_dupfixed() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT | DatabaseFlags::DUP_FIXED).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key1", b"val1", WriteFlags::empty()).unwrap();
        txn.put(db, b"key1", b"val2", WriteFlags::empty()).unwrap();
        txn.put(db, b"key1", b"val3", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val4", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val5", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val6", WriteFlags::empty()).unwrap();

        let cursor = txn.open_ro_cursor(db).unwrap();
        assert_eq!((Some(&b"key1"[..]), &b"val1"[..]), cursor.get(None, None, MDB_FIRST).unwrap());
        assert_eq!((None, &b"val1val2val3"[..]), cursor.get(None, None, MDB_GET_MULTIPLE).unwrap());
        assert!(cursor.get(None, None, MDB_NEXT_MULTIPLE).is_err());
    }

    #[test]
    fn test_iter() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key5", b"val5")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        // Because Result implements FromIterator, we can collect the iterator
        // of items of type Result<_, E> into a Result<Vec<_, E>> by specifying
        // the collection type via the turbofish syntax.
        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter();
            assert_eq!(items, iter.collect::<Result<Vec<_>>>().unwrap());
        }

        // Alternately, we can collect it into an appropriately typed variable.
        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter_start();
            let retr: Result<Vec<_>> = iter.collect();
            assert_eq!(items, retr.unwrap());
        }

        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            cursor.get(Some(b"key2"), None, MDB_SET).unwrap();
            let iter = cursor.into_iter();
            assert_eq!(
                items.clone().into_iter().skip(2).collect::<Vec<_>>(),
                iter.collect::<Result<Vec<_>>>().unwrap()
            );
        }

        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter_start();
            assert_eq!(items, iter.collect::<Result<Vec<_>>>().unwrap());
        }

        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter_from(b"key2");
            let cursor = iter.into_cursor().unwrap();
            assert_eq!(
                items.clone().into_iter().skip(1).collect::<Vec<_>>(),
                cursor.into_iter_from(b"key2").collect::<Result<Vec<_>>>().unwrap()
            );
        }

        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter_from(b"key4");
            assert_eq!(
                items.clone().into_iter().skip(3).collect::<Vec<_>>(),
                iter.collect::<Result<Vec<_>>>().unwrap()
            );
        }

        {
            let cursor = txn.open_ro_cursor(db).unwrap();
            let iter = cursor.into_iter_from(b"key6");
            assert_eq!(vec!().into_iter().collect::<Vec<(&[u8], &[u8])>>(), iter.collect::<Result<Vec<_>>>().unwrap());
        }
    }

    #[test]
    fn test_iter_empty_database() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();
        let txn = env.begin_ro_txn().unwrap();

        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_start().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from(b"foo").count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_rev().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"foo").count());
    }

    #[test]
    fn test_iter_empty_dup_database() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();
        let txn = env.begin_ro_txn().unwrap();

        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_start().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from(b"foo").count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_dup_of(b"foo").count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_rev().count());
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"foo").count());
    }

    #[test]
    fn test_iter_dup() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> = vec![
            (b"a", b"1"),
            (b"a", b"2"),
            (b"a", b"3"),
            (b"b", b"1"),
            (b"b", b"2"),
            (b"b", b"3"),
            (b"c", b"1"),
            (b"c", b"2"),
            (b"c", b"3"),
            (b"e", b"1"),
            (b"e", b"2"),
            (b"e", b"3"),
        ];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        let cursor = txn.open_ro_cursor(db).unwrap();
        assert_eq!(
            items.clone().into_iter().skip(3).take(3).collect::<Vec<(&[u8], &[u8])>>(),
            cursor.into_iter_dup_of(b"b").collect::<Result<Vec<_>>>().unwrap()
        );

        let cursor = txn.open_ro_cursor(db).unwrap();
        assert_eq!(0, cursor.into_iter_dup_of(b"foo").count());
    }

    #[test]
    fn test_iter_rev() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key5", b"val5")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        let forward = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        let backward = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();

        // Reverse-all is exactly the forward iteration reversed.
        assert_eq!(forward.iter().rev().copied().collect::<Vec<_>>(), backward);
        assert_eq!(items.iter().rev().copied().collect::<Vec<_>>(), backward);
    }

    #[test]
    fn test_iter_rev_ignores_position() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key5", b"val5")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        let cursor = txn.open_ro_cursor(db).unwrap();
        // Pre-seeking the cursor to a middle key must not change a full reverse scan.
        cursor.get(Some(b"key2"), None, MDB_SET).unwrap();
        let backward = cursor.into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(items.iter().rev().copied().collect::<Vec<_>>(), backward);
    }

    #[test]
    fn test_iter_from_rev() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key5", b"val5")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();
        let rev_from =
            |key: &[u8]| txn.open_ro_cursor(db).unwrap().into_iter_from_rev(key).collect::<Result<Vec<_>>>().unwrap();

        // Requested key present: included, then descending.
        assert_eq!(vec![items[1], items[0]], rev_from(b"key2"));
        // Requested key present and smallest: only that key.
        assert_eq!(vec![items[0]], rev_from(b"key1"));
        // Requested key absent inside the range: starts at the nearest smaller key.
        assert_eq!(vec![items[2], items[1], items[0]], rev_from(b"key4"));
        // Requested key greater than every key: full descending scan.
        assert_eq!(vec![items[3], items[2], items[1], items[0]], rev_from(b"key6"));
        // Requested key smaller than every key: empty.
        assert_eq!(Vec::<(&[u8], &[u8])>::new(), rev_from(b"key0"));
    }

    #[test]
    fn test_iter_from_rev_map_borrowed_key() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> = vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3")];
        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        // This key slice borrows from the memory map: its pointer is the
        // stored key's address, the natural newest-first pagination pattern.
        // The slice borrows the source cursor, so the cursor is kept in a local
        // for as long as the slice is used.
        let seek_cursor = txn.open_ro_cursor(db).unwrap();
        let (borrowed, _) = seek_cursor.get(Some(b"key2"), None, MDB_SET_KEY).unwrap();
        let borrowed = borrowed.unwrap();

        let out = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(borrowed).collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![items[1], items[0]], out);
    }

    #[test]
    fn test_iter_from_rev_map_borrowed_prefix() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        // "kex" sorts below "key"; "key" (a prefix of "key2") sorts between
        // "kex" and "key2".
        let items: Vec<(&[u8], &[u8])> = vec![(b"kex", b"v0"), (b"key2", b"v2"), (b"key3", b"v3")];
        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        // Borrow "key2" from the map, then take its 3-byte prefix "key". The
        // prefix still aliases the map, but the true landing is the greater
        // key "key2", which must be excluded. The slice borrows the source
        // cursor, so the cursor is kept in a local for as long as it is used.
        let seek_cursor = txn.open_ro_cursor(db).unwrap();
        let (k2, _) = seek_cursor.get(Some(b"key2"), None, MDB_SET_KEY).unwrap();
        let prefix = &k2.unwrap()[..3];

        let out = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(prefix).collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![items[0]], out);
    }

    #[test]
    fn test_iter_from_rev_empty_key_errors() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key", b"val", WriteFlags::empty()).unwrap();
        txn.commit().unwrap();

        let txn = env.begin_ro_txn().unwrap();
        // A zero-length key is rejected by LMDB (MDB_BAD_VALSIZE); the error
        // surfaces on the first step rather than as an empty iterator.
        let mut iter = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"");
        assert!(matches!(iter.next(), Some(Err(_))));
    }

    #[test]
    fn test_iter_rev_prefix_keys() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        // - Keys where one is a byte-prefix of another.
        // - LMDB's default order puts the shorter key first.
        let items: Vec<(&[u8], &[u8])> =
            vec![(&[0x00], b"a"), (&[0x00, 0x00], b"b"), (&[0x00, 0x00, 0x00], b"c"), (&[0x01], b"d")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        let backward = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(items.iter().rev().copied().collect::<Vec<_>>(), backward);

        let rev_from =
            |key: &[u8]| txn.open_ro_cursor(db).unwrap().into_iter_from_rev(key).collect::<Result<Vec<_>>>().unwrap();

        // Exact prefix key: its longer extensions are excluded.
        assert_eq!(vec![items[1], items[0]], rev_from(&[0x00, 0x00]));
        // Absent key inside the prefix chain: starts below it.
        assert_eq!(vec![items[2], items[1], items[0]], rev_from(&[0x00, 0x00, 0x01]));
        // Exact smallest key: only itself.
        assert_eq!(vec![items[0]], rev_from(&[0x00]));
    }

    #[test]
    fn test_iter_rev_single_key() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key", b"val", WriteFlags::empty()).unwrap();
        txn.commit().unwrap();

        let txn = env.begin_ro_txn().unwrap();
        let expected: Vec<(&[u8], &[u8])> = vec![(b"key", b"val")];

        assert_eq!(expected, txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap());
        assert_eq!(
            expected,
            txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"key").collect::<Result<Vec<_>>>().unwrap()
        );
        // A request above the only key still finds it.
        assert_eq!(
            expected,
            txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"kez").collect::<Result<Vec<_>>>().unwrap()
        );
        // A request below the only key finds nothing.
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"kex").count());
    }

    #[test]
    fn test_iter_rev_dup() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"a", b"1"), (b"a", b"2"), (b"a", b"3"), (b"b", b"1"), (b"b", b"2"), (b"b", b"3")];

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        // Reverse-all mirrors the forward iteration exactly, duplicates included.
        let backward = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(items.iter().rev().copied().collect::<Vec<_>>(), backward);

        // Exact hit now yields every duplicate of the matched key, descending,
        // before crossing to earlier keys.
        let from_b = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"b").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(
            vec![(&b"b"[..], &b"3"[..]), (b"b", b"2"), (b"b", b"1"), (b"a", b"3"), (b"a", b"2"), (b"a", b"1")],
            from_b
        );

        // A key greater than all keys seeds MDB_LAST and yields the same set.
        let from_c = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"c").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(from_b, from_c);

        // Absent key between a and b: a's duplicates only.
        let from_ab = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"ab").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![(&b"a"[..], &b"3"[..]), (b"a", b"2"), (b"a", b"1")], from_ab);

        // A key below all keys: empty.
        assert_eq!(0, txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"0").count());
    }

    #[test]
    fn test_iter_from_rev_dup_middle_key() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> = vec![
            (b"a", b"1"),
            (b"a", b"2"),
            (b"a", b"3"),
            (b"b", b"1"),
            (b"b", b"2"),
            (b"b", b"3"),
            (b"c", b"1"),
            (b"c", b"2"),
            (b"c", b"3"),
        ];
        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let txn = env.begin_ro_txn().unwrap();

        // Exact hit on the middle key "b" yields all of b's duplicates
        // descending, then a's; "c" and its duplicates are excluded.
        let from_b = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"b").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(
            vec![(&b"b"[..], &b"3"[..]), (b"b", b"2"), (b"b", b"1"), (b"a", b"3"), (b"a", b"2"), (b"a", b"1")],
            from_b
        );

        // Exact hit on the smallest key "a" yields only a's duplicates, then
        // the descending walk underflows to an empty result.
        let from_a = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"a").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![(&b"a"[..], &b"3"[..]), (b"a", b"2"), (b"a", b"1")], from_a);
    }

    #[test]
    fn test_iter_from_rev_dup_subdata_spill() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        // - Roughly 2000 eight-byte duplicates under one key dwarf the node
        //   size limit and force the duplicates into a separate sub-database
        //   (F_SUBDATA) rather than an inline sub-page.
        // - A neighbor key verifies the descending walk crosses out of the
        //   sub-database back into the main database.
        let b_vals: Vec<[u8; 8]> = (0u64..2000).map(u64::to_be_bytes).collect();
        let a_vals: Vec<[u8; 8]> = vec![1u64.to_be_bytes(), 2u64.to_be_bytes()];
        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for v in &a_vals {
                txn.put(db, b"a", v, WriteFlags::empty()).unwrap();
            }
            for v in &b_vals {
                txn.put(db, b"b", v, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }
        let txn = env.begin_ro_txn().unwrap();

        let mut expected: Vec<(&[u8], &[u8])> = Vec::new();
        for v in b_vals.iter().rev() {
            expected.push((b"b", &v[..]));
        }
        for v in a_vals.iter().rev() {
            expected.push((b"a", &v[..]));
        }

        let from_b = txn.open_ro_cursor(db).unwrap().into_iter_from_rev(b"b").collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(expected, from_b);

        let full = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(expected, full);
    }

    #[test]
    fn test_iter_rev_randomized() {
        // Seed is printed so a failing run can be replayed by hardcoding it.
        let seed: u64 = rand::random();
        println!("test_iter_rev_randomized seed: {}", seed);
        // Replaying a hardcoded seed is valid only while rand stays pinned to
        // 0.10; StdRng's algorithm is not stable across rand major versions.
        let mut rng = StdRng::seed_from_u64(seed);

        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        // The oracle orders byte keys the same way LMDB's default comparator
        // does: lexicographic, with a shorter prefix sorting first.
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();

        let mut txn = env.begin_rw_txn(None).unwrap();
        for _ in 0..500 {
            let key: Vec<u8> = (0..rng.random_range(1..=32)).map(|_| rng.random()).collect();
            let value: Vec<u8> = (0..rng.random_range(0..=16)).map(|_| rng.random()).collect();
            txn.put(db, &key, &value, WriteFlags::empty()).unwrap();
            oracle.insert(key, value);
        }
        txn.commit().unwrap();

        let txn = env.begin_ro_txn().unwrap();

        let expected: Vec<(&[u8], &[u8])> =
            oracle.iter().rev().map(|(key, value)| (key.as_slice(), value.as_slice())).collect();
        let backward = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(expected, backward, "seed {}", seed);

        // Random probes: half drawn from present keys, half fresh and mostly absent.
        for _ in 0..64 {
            let probe: Vec<u8> = if rng.random_bool(0.5) {
                let index = rng.random_range(0..oracle.len());
                oracle.keys().nth(index).unwrap().clone()
            } else {
                (0..rng.random_range(1..=32)).map(|_| rng.random()).collect()
            };
            let expected: Vec<(&[u8], &[u8])> =
                oracle.range(..=probe.clone()).rev().map(|(key, value)| (key.as_slice(), value.as_slice())).collect();
            let actual =
                txn.open_ro_cursor(db).unwrap().into_iter_from_rev(&probe).collect::<Result<Vec<_>>>().unwrap();
            assert_eq!(expected, actual, "seed {} probe {:?}", seed, probe);
        }
    }

    #[test]
    fn test_iter_rev_dup_randomized() {
        // Seed is printed so a failing run can be replayed by hardcoding it.
        let seed: u64 = rand::random();
        println!("test_iter_rev_dup_randomized seed: {}", seed);
        // Replaying a hardcoded seed is valid only while rand stays pinned to
        // 0.10; StdRng's algorithm is not stable across rand major versions.
        let mut rng = StdRng::seed_from_u64(seed);

        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        // - The oracle mirrors LMDB's DUP_SORT order: keys ascending, and within
        //   a key the duplicate values ascending, both lexicographic.
        // - A BTreeSet of values drops exact duplicates, matching a plain
        //   DUP_SORT put that silently ignores an already-present (key, value).
        let mut oracle: BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>> = BTreeMap::new();

        let mut txn = env.begin_rw_txn(None).unwrap();
        for _ in 0..300 {
            let key: Vec<u8> = (0..rng.random_range(1..=32)).map(|_| rng.random()).collect();
            // DUP_SORT stores each duplicate value as a key in a sub-database,
            // and LMDB rejects empty keys, so a duplicate value must be at least
            // one byte. Zero-length values cannot round-trip here.
            let value: Vec<u8> = (0..rng.random_range(1..=16)).map(|_| rng.random()).collect();
            txn.put(db, &key, &value, WriteFlags::empty()).unwrap();
            oracle.entry(key).or_default().insert(value);
        }
        txn.commit().unwrap();

        let txn = env.begin_ro_txn().unwrap();

        // Full reverse: each key descending, and within a key each duplicate
        // value descending.
        let expected: Vec<(&[u8], &[u8])> = oracle
            .iter()
            .rev()
            .flat_map(|(key, values)| values.iter().rev().map(move |value| (key.as_slice(), value.as_slice())))
            .collect();
        let backward = txn.open_ro_cursor(db).unwrap().into_iter_rev().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(expected, backward, "seed {}", seed);

        // Random probes: half drawn from present keys, half fresh and mostly absent.
        for _ in 0..32 {
            let probe: Vec<u8> = if rng.random_bool(0.5) {
                let index = rng.random_range(0..oracle.len());
                oracle.keys().nth(index).unwrap().clone()
            } else {
                (0..rng.random_range(1..=32)).map(|_| rng.random()).collect()
            };
            let expected: Vec<(&[u8], &[u8])> = oracle
                .range(..=probe.clone())
                .rev()
                .flat_map(|(key, values)| values.iter().rev().map(move |value| (key.as_slice(), value.as_slice())))
                .collect();
            let actual =
                txn.open_ro_cursor(db).unwrap().into_iter_from_rev(&probe).collect::<Result<Vec<_>>>().unwrap();
            assert_eq!(expected, actual, "seed {} probe {:?}", seed, probe);
        }
    }

    #[test]
    fn test_iter_del_get() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> = vec![(b"a", b"1"), (b"b", b"2")];
        let r: Vec<(&[u8], &[u8])> = Vec::new();
        {
            let txn = env.begin_ro_txn().unwrap();
            let cursor = txn.open_ro_cursor(db).unwrap();
            assert_eq!(r, cursor.into_iter_dup_of(b"a").collect::<Result<Vec<_>>>().unwrap());
        }

        {
            let mut txn = env.begin_rw_txn(None).unwrap();
            for (key, data) in &items {
                txn.put(db, key, data, WriteFlags::empty()).unwrap();
            }
            txn.commit().unwrap();
        }

        let mut txn = env.begin_rw_txn(None).unwrap();

        let cursor = txn.open_rw_cursor(db).unwrap();
        assert_eq!(
            items.clone().into_iter().take(1).collect::<Vec<(&[u8], &[u8])>>(),
            cursor.into_iter_dup_of(b"a").collect::<Result<Vec<_>>>().unwrap()
        );

        let mut cursor = txn.open_rw_cursor(db).unwrap();
        assert_eq!((None, &b"1"[..]), cursor.get(Some(b"a"), Some(b"1"), MDB_SET).unwrap());
        cursor.del(WriteFlags::empty()).unwrap();

        assert_eq!(r, cursor.into_iter_dup_of(b"a").collect::<Result<Vec<_>>>().unwrap());
    }

    #[test]
    fn test_put_del() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        let mut cursor = txn.open_rw_cursor(db).unwrap();

        cursor.put(b"key1", b"val1", WriteFlags::empty()).unwrap();
        cursor.put(b"key2", b"val2", WriteFlags::empty()).unwrap();
        cursor.put(b"key3", b"val3", WriteFlags::empty()).unwrap();

        assert_eq!((Some(&b"key3"[..]), &b"val3"[..]), cursor.get(None, None, MDB_GET_CURRENT).unwrap());

        cursor.del(WriteFlags::empty()).unwrap();
        assert_eq!((Some(&b"key2"[..]), &b"val2"[..]), cursor.get(None, None, MDB_LAST).unwrap());
    }

    #[test]
    fn test_rw_cursor_walk_fixed() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        txn.put(db, b"key1", b"val1", WriteFlags::empty()).unwrap();
        txn.put(db, b"key2", b"val2", WriteFlags::empty()).unwrap();
        txn.put(db, b"key3", b"val3", WriteFlags::empty()).unwrap();

        let cursor = txn.open_rw_cursor(db).unwrap();

        // An unpositioned cursor starts the forward walk at the first item.
        assert_eq!(Some((&b"key1"[..], &b"val1"[..])), cursor.next().unwrap());
        assert_eq!(Some((&b"key2"[..], &b"val2"[..])), cursor.next().unwrap());
        assert_eq!(Some((&b"key3"[..], &b"val3"[..])), cursor.next().unwrap());
        assert_eq!(None, cursor.next().unwrap());

        assert_eq!(Some((&b"key1"[..], &b"val1"[..])), cursor.first().unwrap());
        assert_eq!(Some((&b"key3"[..], &b"val3"[..])), cursor.last().unwrap());

        assert_eq!(Some((&b"key2"[..], &b"val2"[..])), cursor.prev().unwrap());
        assert_eq!(Some((&b"key1"[..], &b"val1"[..])), cursor.prev().unwrap());
        assert_eq!(None, cursor.prev().unwrap());

        // Moving is not an update operation, so items from separate positioning
        // calls coexist. That is what the shared borrow buys.
        let first = cursor.first().unwrap().unwrap();
        let second = cursor.next().unwrap().unwrap();
        assert_eq!((&b"key1"[..], &b"val1"[..]), first);
        assert_eq!((&b"key2"[..], &b"val2"[..]), second);
    }

    #[test]
    fn test_rw_cursor_walk_empty_database() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let mut txn = env.begin_rw_txn(None).unwrap();
        let cursor = txn.open_rw_cursor(db).unwrap();

        // An empty database ends the walk rather than erroring.
        assert_eq!(None, cursor.first().unwrap());
        assert_eq!(None, cursor.last().unwrap());
        assert_eq!(None, cursor.next().unwrap());
        assert_eq!(None, cursor.prev().unwrap());
    }

    #[test]
    fn test_rw_cursor_walk_dup() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"a", b"1"), (b"a", b"2"), (b"a", b"3"), (b"b", b"1"), (b"b", b"2"), (b"b", b"3")];

        let mut txn = env.begin_rw_txn(None).unwrap();
        for (key, data) in &items {
            txn.put(db, key, data, WriteFlags::empty()).unwrap();
        }

        let cursor = txn.open_rw_cursor(db).unwrap();
        let expected: Vec<(Vec<u8>, Vec<u8>)> = items.iter().map(|(key, data)| (key.to_vec(), data.to_vec())).collect();

        // Every duplicate of a key is walked before crossing to the next key.
        let mut forward: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut step = cursor.first().unwrap();
        while let Some((key, data)) = step {
            forward.push((key.to_vec(), data.to_vec()));
            step = cursor.next().unwrap();
        }
        assert_eq!(expected, forward);

        let mut backward: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        let mut step = cursor.last().unwrap();
        while let Some((key, data)) = step {
            backward.push((key.to_vec(), data.to_vec()));
            step = cursor.prev().unwrap();
        }
        assert_eq!(expected.iter().rev().cloned().collect::<Vec<_>>(), backward);
    }

    #[test]
    fn test_rw_cursor_walk_and_delete() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key4", b"val4"), (b"key5", b"val5")];

        let mut txn = env.begin_rw_txn(None).unwrap();
        for (key, data) in &items {
            txn.put(db, key, data, WriteFlags::empty()).unwrap();
        }

        // Decide while the item is borrowed, then mutate. The borrow ends at the
        // last use of the item, which is what lets `del` take `&mut` next.
        // Holding the item across the `del` is rejected instead.
        {
            let mut cursor = txn.open_rw_cursor(db).unwrap();
            while let Some((key, data)) = cursor.next().unwrap() {
                let doomed = key == &b"key2"[..] || data == &b"val4"[..];
                if doomed {
                    cursor.del(WriteFlags::empty()).unwrap();
                }
            }
        }

        // Exactly the matching entries are gone; the rest are untouched.
        let remaining = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![items[0], items[2], items[4]], remaining);
    }

    #[test]
    fn test_rw_cursor_next_after_del_does_not_skip() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"key1", b"val1"), (b"key2", b"val2"), (b"key3", b"val3"), (b"key4", b"val4"), (b"key5", b"val5")];

        let mut txn = env.begin_rw_txn(None).unwrap();
        for (key, data) in &items {
            txn.put(db, key, data, WriteFlags::empty()).unwrap();
        }

        let mut visited: Vec<Vec<u8>> = Vec::new();
        {
            let mut cursor = txn.open_rw_cursor(db).unwrap();
            while let Some((key, _data)) = cursor.next().unwrap() {
                visited.push(key.to_vec());
                let doomed = key == &b"key3"[..];
                if doomed {
                    cursor.del(WriteFlags::empty()).unwrap();
                }
            }
        }

        // Deleting key3 must not make the following next() skip key4: once
        // key3's slot is closed up the cursor already denotes key4. A vendored
        // LMDB that dropped that behavior would show up here as a missing key4.
        let expected_visited: Vec<Vec<u8>> = items.iter().map(|(key, _data)| key.to_vec()).collect();
        assert_eq!(expected_visited, visited);

        let remaining = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![items[0], items[1], items[3], items[4]], remaining);
    }

    #[test]
    fn test_rw_cursor_walk_and_delete_dup() {
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        let items: Vec<(&[u8], &[u8])> =
            vec![(b"a", b"1"), (b"a", b"2"), (b"a", b"3"), (b"b", b"1"), (b"b", b"2"), (b"b", b"3")];

        let mut txn = env.begin_rw_txn(None).unwrap();
        for (key, data) in &items {
            txn.put(db, key, data, WriteFlags::empty()).unwrap();
        }

        let mut visited: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        {
            let mut cursor = txn.open_rw_cursor(db).unwrap();
            while let Some((key, data)) = cursor.next().unwrap() {
                visited.push((key.to_vec(), data.to_vec()));
                let doomed = data == &b"2"[..];
                if doomed {
                    cursor.del(WriteFlags::empty()).unwrap();
                }
            }
        }

        // Deleting a duplicate must not cost the visit of the one after it,
        // which exercises the same position contract inside a duplicate run.
        let expected_visited: Vec<(Vec<u8>, Vec<u8>)> =
            items.iter().map(|(key, data)| (key.to_vec(), data.to_vec())).collect();
        assert_eq!(expected_visited, visited);

        let remaining = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        assert_eq!(vec![items[0], items[2], items[3], items[5]], remaining);
    }

    #[test]
    fn test_rw_cursor_walk_and_delete_randomized() {
        // Seed is printed so a failing run can be replayed by hardcoding it.
        let seed: u64 = rand::random();
        println!("test_rw_cursor_walk_and_delete_randomized seed: {}", seed);
        // Replaying a hardcoded seed is valid only while rand stays pinned to
        // 0.10; StdRng's algorithm is not stable across rand major versions.
        let mut rng = StdRng::seed_from_u64(seed);

        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.open_db(None).unwrap();

        // The oracle orders byte keys the same way LMDB's default comparator
        // does: lexicographic, with a shorter prefix sorting first.
        let mut oracle: BTreeMap<Vec<u8>, Vec<u8>> = BTreeMap::new();

        let mut txn = env.begin_rw_txn(None).unwrap();
        let count: usize = rng.random_range(1..=400);
        for _ in 0..count {
            let key: Vec<u8> = (0..rng.random_range(1..=32)).map(|_| rng.random()).collect();
            let value: Vec<u8> = (0..rng.random_range(0..=16)).map(|_| rng.random()).collect();
            txn.put(db, &key, &value, WriteFlags::empty()).unwrap();
            oracle.insert(key, value);
        }

        // A random threshold on the key's first byte drives the deletions, so a
        // run covers lone deletes, runs of consecutive deletes, and deletes at
        // either end.
        let threshold: u8 = rng.random();
        let doomed = |key: &[u8]| match key.first() {
            Some(byte) => *byte < threshold,
            None => false,
        };

        let mut visited: Vec<Vec<u8>> = Vec::new();
        {
            let mut cursor = txn.open_rw_cursor(db).unwrap();
            while let Some((key, _data)) = cursor.next().unwrap() {
                visited.push(key.to_vec());
                if doomed(key) {
                    cursor.del(WriteFlags::empty()).unwrap();
                }
            }
        }

        // No delete may cost a visit: every stored key is seen exactly once, in
        // order, however the deletions fall.
        let expected_visited: Vec<Vec<u8>> = oracle.keys().cloned().collect();
        assert_eq!(expected_visited, visited, "seed {}", seed);

        oracle.retain(|key, _value| !doomed(key));

        let remaining = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        let expected: Vec<(&[u8], &[u8])> =
            oracle.iter().map(|(key, value)| (key.as_slice(), value.as_slice())).collect();
        assert_eq!(expected, remaining, "seed {}", seed);
    }

    #[test]
    fn test_rw_cursor_walk_and_delete_dup_randomized() {
        // Seed is printed so a failing run can be replayed by hardcoding it.
        let seed: u64 = rand::random();
        println!("test_rw_cursor_walk_and_delete_dup_randomized seed: {}", seed);
        // Replaying a hardcoded seed is valid only while rand stays pinned to
        // 0.10; StdRng's algorithm is not stable across rand major versions.
        let mut rng = StdRng::seed_from_u64(seed);

        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().open(dir.path()).unwrap();
        let db = env.create_db(None, DatabaseFlags::DUP_SORT).unwrap();

        // - The oracle mirrors LMDB's DUP_SORT order: keys ascending, and within
        //   a key the duplicate values ascending, both lexicographic.
        // - A BTreeSet of values drops exact duplicates, matching a plain
        //   DUP_SORT put that silently ignores an already-present (key, value).
        let mut oracle: BTreeMap<Vec<u8>, BTreeSet<Vec<u8>>> = BTreeMap::new();

        let mut txn = env.begin_rw_txn(None).unwrap();
        let count: usize = rng.random_range(1..=300);
        for _ in 0..count {
            // A small key alphabet piles duplicates under each key instead of
            // making every key unique.
            let key: Vec<u8> = vec![rng.random_range(b'a'..=b'e')];
            // DUP_SORT stores each duplicate value as a key in a sub-database,
            // and LMDB rejects empty keys, so a duplicate value must be at least
            // one byte.
            let value: Vec<u8> = (0..rng.random_range(1..=16)).map(|_| rng.random()).collect();
            txn.put(db, &key, &value, WriteFlags::empty()).unwrap();
            oracle.entry(key).or_default().insert(value);
        }

        // The predicate reads the duplicate value, so deletions land inside a
        // key's duplicate run as well as at its edges.
        let threshold: u8 = rng.random();
        let doomed = |value: &[u8]| match value.first() {
            Some(byte) => *byte < threshold,
            None => false,
        };

        let mut visited: Vec<(Vec<u8>, Vec<u8>)> = Vec::new();
        {
            let mut cursor = txn.open_rw_cursor(db).unwrap();
            while let Some((key, data)) = cursor.next().unwrap() {
                visited.push((key.to_vec(), data.to_vec()));
                if doomed(data) {
                    cursor.del(WriteFlags::empty()).unwrap();
                }
            }
        }

        // Deleting a duplicate must not cost the visit of the one after it.
        let expected_visited: Vec<(Vec<u8>, Vec<u8>)> = oracle
            .iter()
            .flat_map(|(key, values)| values.iter().map(move |value| (key.clone(), value.clone())))
            .collect();
        assert_eq!(expected_visited, visited, "seed {}", seed);

        for values in oracle.values_mut() {
            values.retain(|value| !doomed(value));
        }

        let remaining = txn.open_ro_cursor(db).unwrap().into_iter_start().collect::<Result<Vec<_>>>().unwrap();
        let expected: Vec<(&[u8], &[u8])> = oracle
            .iter()
            .flat_map(|(key, values)| values.iter().map(move |value| (key.as_slice(), value.as_slice())))
            .collect();
        assert_eq!(expected, remaining, "seed {}", seed);
    }
}
