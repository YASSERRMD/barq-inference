//! Structured logging for production observability
//!
//! Provides JSON-formatted structured logging with configurable levels,
//! request correlation IDs, and integration with metrics system.

use std::time::{SystemTime, UNIX_EPOCH};
use serde::{Deserialize, Serialize};
use std::sync::Arc;
use tokio::sync::RwLock;

/// Log level
#[derive(Debug, Clone, Copy, PartialEq, Eq, PartialOrd, Ord)]
pub enum LogLevel {
    Debug = 0,
    Info = 1,
    Warn = 2,
    Error = 3,
}

impl LogLevel {
    pub fn as_str(&self) -> &'static str {
        match self {
            LogLevel::Debug => "DEBUG",
            LogLevel::Info => "INFO",
            LogLevel::Warn => "WARN",
            LogLevel::Error => "ERROR",
        }
    }
}

/// Structured log entry
#[derive(Debug, Clone, Serialize, Deserialize)]
pub struct LogEntry {
    /// Timestamp (Unix milliseconds)
    pub timestamp: u64,
    /// Log level
    pub level: String,
    /// Log message
    pub message: String,
    /// Request ID for correlation
    #[serde(skip_serializing_if = "Option::is_none")]
    pub request_id: Option<String>,
    /// Module path
    #[serde(skip_serializing_if = "Option::is_none")]
    pub module: Option<String>,
    /// Additional context
    #[serde(skip_serializing_if = "Option::is_none")]
    pub context: Option<serde_json::Value>,
}

/// Logging configuration
#[derive(Debug, Clone)]
pub struct LoggingConfig {
    /// Minimum log level
    pub min_level: LogLevel,
    /// Enable JSON formatting
    pub json_format: bool,
    /// Enable request correlation
    pub enable_correlation: bool,
    /// Log to stdout
    pub log_to_stdout: bool,
    /// Log to file
    pub log_file: Option<String>,
}

impl Default for LoggingConfig {
    fn default() -> Self {
        Self {
            min_level: LogLevel::Info,
            json_format: true,
            enable_correlation: true,
            log_to_stdout: true,
            log_file: None,
        }
    }
}

impl LoggingConfig {
    /// Development-friendly config with colored output
    pub fn development() -> Self {
        Self {
            min_level: LogLevel::Debug,
            json_format: false,
            enable_correlation: true,
            log_to_stdout: true,
            log_file: None,
        }
    }

    /// Production config with JSON output
    pub fn production() -> Self {
        Self {
            min_level: LogLevel::Info,
            json_format: true,
            enable_correlation: true,
            log_to_stdout: true,
            log_file: Some("/var/log/barq-inference.log".to_string()),
        }
    }
}

/// Structured logger
pub struct Logger {
    config: LoggingConfig,
}

impl Logger {
    pub fn new(config: LoggingConfig) -> Self {
        Self { config }
    }

    pub fn log(&self, level: LogLevel, message: &str, context: Option<serde_json::Value>) {
        if level < self.config.min_level {
            return;
        }

        let entry = LogEntry {
            timestamp: current_timestamp_ms(),
            level: level.as_str().to_string(),
            message: message.to_string(),
            request_id: None,
            module: None,
            context,
        };

        if self.config.json_format {
            if let Ok(json) = serde_json::to_string(&entry) {
                println!("{}", json);
            }
        } else {
            println!("[{}] {}", entry.level, entry.message);
        }
    }

    pub fn debug(&self, message: &str) {
        self.log(LogLevel::Debug, message, None);
    }

    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message, None);
    }

    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message, None);
    }

    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message, None);
    }
}

/// Request-scoped logger with correlation ID
pub struct RequestLogger {
    request_id: String,
    logger: Arc<Logger>,
}

impl RequestLogger {
    pub fn new(request_id: String, logger: Arc<Logger>) -> Self {
        Self { request_id, logger }
    }

    pub fn log(&self, level: LogLevel, message: &str, context: Option<serde_json::Value>) {
        if level < self.logger.config.min_level {
            return;
        }

        let mut entry = LogEntry {
            timestamp: current_timestamp_ms(),
            level: level.as_str().to_string(),
            message: message.to_string(),
            request_id: Some(self.request_id.clone()),
            module: None,
            context,
        };

        if self.logger.config.json_format {
            if let Ok(json) = serde_json::to_string(&entry) {
                println!("{}", json);
            }
        } else {
            println!(
                "[{}][request_id={}] {}",
                entry.level,
                self.request_id,
                entry.message
            );
        }
    }

    pub fn info(&self, message: &str) {
        self.log(LogLevel::Info, message, None);
    }

    pub fn warn(&self, message: &str) {
        self.log(LogLevel::Warn, message, None);
    }

    pub fn error(&self, message: &str) {
        self.log(LogLevel::Error, message, None);
    }
}

/// Get current timestamp in milliseconds
fn current_timestamp_ms() -> u64 {
    SystemTime::now()
        .duration_since(UNIX_EPOCH)
        .unwrap()
        .as_millis() as u64
}

/// Global logger instance
static GLOBAL_LOGGER: std::sync::OnceLock<Arc<Logger>> = std::sync::OnceLock::new();

/// Initialize global logger
pub fn init_logger(config: LoggingConfig) {
    let logger = Arc::new(Logger::new(config));
    GLOBAL_LOGGER.set(logger).ok();
}

/// Get global logger
pub fn logger() -> Option<Arc<Logger>> {
    GLOBAL_LOGGER.get().cloned()
}

/// Convenience function for logging
pub fn log_info(message: &str) {
    if let Some(logger) = logger() {
        logger.info(message);
    }
}

pub fn log_warn(message: &str) {
    if let Some(logger) = logger() {
        logger.warn(message);
    }
}

pub fn log_error(message: &str) {
    if let Some(logger) = logger() {
        logger.error(message);
    }
}

pub fn log_debug(message: &str) {
    if let Some(logger) = logger() {
        logger.debug(message);
    }
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn test_log_level_comparison() {
        assert!(LogLevel::Debug < LogLevel::Info);
        assert!(LogLevel::Info < LogLevel::Warn);
        assert!(LogLevel::Warn < LogLevel::Error);
    }

    #[test]
    fn test_logging_config_default() {
        let config = LoggingConfig::default();
        assert_eq!(config.min_level, LogLevel::Info);
        assert!(config.json_format);
    }

    #[test]
    fn test_logging_config_development() {
        let config = LoggingConfig::development();
        assert_eq!(config.min_level, LogLevel::Debug);
        assert!(!config.json_format);
    }

    #[test]
    fn test_logging_config_production() {
        let config = LoggingConfig::production();
        assert_eq!(config.min_level, LogLevel::Info);
        assert!(config.json_format);
        assert!(config.log_file.is_some());
    }

    #[test]
    fn test_logger_creation() {
        let logger = Logger::new(LoggingConfig::default());
        // Should not panic
        logger.info("Test message");
    }

    #[test]
    fn test_request_logger() {
        let logger = Arc::new(Logger::new(LoggingConfig::default()));
        let request_logger = RequestLogger::new("test-123".to_string(), logger);
        // Should not panic
        request_logger.info("Test message");
    }
}
