
use std::borrow::Borrow;
use std::sync::{OnceLock, RwLock};

pub type LogFn = Box<dyn Fn(&str) -> () + Send + Sync>;

struct Logger {
    log_fns: RwLock<Vec<LogFn>>
}

impl Default for Logger {
    fn default() -> Self {
        Self { log_fns: RwLock::new(vec![]) }
    }
}

impl Logger {
    fn subscribe(&self, log_f: LogFn) {
        let mut log_fns = self.log_fns.write().expect("Log lock poisoned");
        log_fns.push(log_f);
    }

    fn log(&self, s: &str) {
        for log_fn in self.log_fns.read().expect("Log lock poisoned").iter() {
            log_fn(s);
        }
    }
}

static LOGGER: OnceLock<Logger> = OnceLock::new();

pub fn subscribe(log_f: LogFn) {
    LOGGER.get_or_init(Logger::default).subscribe(log_f)
}

pub fn log<S: Borrow<str>>(s: S) {
    LOGGER.get_or_init(Logger::default).log(s.borrow())
}