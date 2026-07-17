use std::sync::atomic::Ordering;

use crate::Environment;

#[must_use = "TransactionGuard has no effect without holding its object"]
pub struct TransactionGuard<'a> {
    env: &'a Environment,
}

/// A transaction guard is a global counter that tracks all transactions currently open.
/// It lets operations that must exclude live transactions, such as a map resize, wait for them to close.
impl<'a> TransactionGuard<'a> {
    pub fn new(env: &'a Environment) -> Self {
        // - reader side of a lock-free handshake with the resizer (ScopedTransactionBlocker)
        // - publish "a transaction is going live" by bumping tx_count FIRST, then check if a resize is pending
        // - the store of tx_count and the load of tx_blocker_count are the store-buffer (Dekker) pattern:
        //   the resizer does the mirror image (store tx_blocker_count, then load tx_count)
        // - the one outcome that must be impossible is the reader's load missing the resizer's store while
        //   the resizer's load misses the reader's store; that would let a reader go live on a mapping being
        //   remapped (use-after-munmap). Forbidding it needs StoreLoad ordering, which only SeqCst provides;
        //   AcqRel would silently reintroduce the race, so both store-then-load pairs stay SeqCst.
        loop {
            env.tx_count().fetch_add(1, Ordering::SeqCst);
            if env.tx_blocker_count().load(Ordering::SeqCst) == 0 {
                break;
            }
            // - a resize is pending: back our count out so the resizer can drain to zero
            env.tx_count().fetch_sub(1, Ordering::SeqCst);
            // - wait for the blocker to clear before retrying; this is only a backoff poll, so Relaxed suffices
            while env.tx_blocker_count().load(Ordering::Relaxed) != 0 {
                std::thread::yield_now();
            }
        }
        Self {
            env,
        }
    }

    pub fn wait_for_transactions_to_finish(env: &'a Environment) {
        // - resizer side: block until every live transaction has dropped its guard
        // - this load is the load half of the resizer's (store tx_blocker, then load tx_count) pair, so it
        //   must be SeqCst: with a weaker load the resizer could reorder it ahead of its blocker store and
        //   miss a reader's tx_count increment while that reader misses the blocker store, letting the reader
        //   go live on a map about to be remapped (the store-buffer race; see TransactionGuard::new)
        // - SeqCst also subsumes Acquire, so it still pairs with each guard's Release decrement, making the
        //   finished transactions' effects visible before the remap
        while env.tx_count().load(Ordering::SeqCst) != 0 {
            std::thread::yield_now();
        }
    }
}

impl<'a> Drop for TransactionGuard<'a> {
    fn drop(&mut self) {
        // - Release so a resizer that later observes tx_count == 0 also sees this transaction's effects
        self.env.tx_count().fetch_sub(1, Ordering::Release);
    }
}

#[must_use = "ScopedTransactionBlocker has no effect without holding its object"]
pub struct ScopedTransactionBlocker<'a> {
    env: &'a Environment,
}

/// A ScopedTransactionBlocker prevents new transactions from going live.
/// While an instance is alive, TransactionGuard::new spins in its backoff loop instead of returning a guard.
impl<'a> ScopedTransactionBlocker<'a> {
    pub fn new(env: &'a Environment) -> Self {
        // - resizer side of the handshake: publish "resize pending" FIRST, before draining tx_count
        // - this store, paired with the tx_count load in wait_for_transactions_to_finish, is the mirror of the
        //   reader's store-then-load; SeqCst on both pairs is what forbids the racing interleaving (see
        //   TransactionGuard::new for the full store-buffer rationale)
        env.tx_blocker_count().fetch_add(1, Ordering::SeqCst);
        Self {
            env,
        }
    }
}

impl<'a> Drop for ScopedTransactionBlocker<'a> {
    fn drop(&mut self) {
        // - Release so readers that resume after the blocker clears observe the completed resize
        self.env.tx_blocker_count().fetch_sub(1, Ordering::Release);
    }
}
