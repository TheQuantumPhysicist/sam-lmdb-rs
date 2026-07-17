use libc::c_uint;
use std::ffi::CString;
use std::ptr;

use lmdb_sys as ffi;

use crate::error::{lmdb_result, Error, Result};

/// A handle to an individual database in an environment.
///
/// A database handle denotes the name and parameters of a database in an environment.
#[derive(Clone, Copy, Debug, Eq, PartialEq)]
pub struct Database {
    dbi: ffi::MDB_dbi,
}

impl Database {
    /// Opens a new database handle in the given transaction.
    ///
    /// Prefer using `Environment::open_db`, `Environment::create_db`, `TransactionExt::open_db`,
    /// or `RwTransaction::create_db`.
    pub(crate) unsafe fn new(txn: *mut ffi::MDB_txn, name: Option<&str>, flags: c_uint) -> Result<Database> {
        // A database name is caller-supplied input, so an interior NUL byte is a
        // recoverable bad-input error, not an invariant violation. Mirror the
        // env-open path, which rejects a NUL-bearing path with Error::Invalid.
        let c_name = match name {
            Some(n) => match CString::new(n) {
                Ok(c_name) => Some(c_name),
                Err(..) => return Err(Error::Invalid),
            },
            None => None,
        };
        let name_ptr = if let Some(ref c_name) = c_name {
            c_name.as_ptr()
        } else {
            ptr::null()
        };
        let mut dbi: ffi::MDB_dbi = 0;
        lmdb_result(ffi::mdb_dbi_open(txn, name_ptr, flags, &mut dbi))?;
        Ok(Database {
            dbi,
        })
    }

    pub(crate) fn freelist_db() -> Database {
        Database {
            dbi: 0,
        }
    }

    /// Returns the underlying LMDB database handle.
    ///
    /// The caller **must** ensure that the handle is not used after the lifetime of the
    /// environment, or after the database has been closed.
    #[allow(clippy::trivially_copy_pass_by_ref)]
    pub fn dbi(&self) -> ffi::MDB_dbi {
        self.dbi
    }
}

#[cfg(test)]
mod test {
    use super::*;
    use crate::environment::Environment;
    use crate::flags::DatabaseFlags;
    use tempdir::TempDir;

    // Just to check this compiles
    #[allow(unused)]
    fn database_is_send_sync(db: Database) {
        fn is_send_sync(_x: impl Send + Sync) {}
        is_send_sync(db)
    }

    #[test]
    fn test_create_db_with_interior_nul_returns_err() {
        // A DB name is caller input; an interior NUL must be a recoverable
        // error, not a panic. Reaching Database::new through the public
        // create_db path must return Err, matching the env-open path.
        let dir = TempDir::new("test").unwrap();
        let env = Environment::new().set_max_dbs(1).open(dir.path()).unwrap();

        let result = env.create_db(Some("bad\0name"), DatabaseFlags::empty());
        assert_eq!(result, Err(Error::Invalid));
    }
}
