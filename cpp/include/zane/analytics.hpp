// Zane Analytics — the C++ "analytical mode" core of Zane's digital mind.
//
// This engine owns the numeric side of Zane's personality: the exact,
// pseudo-randomized success-probability readouts he quotes mid-mission,
// and the low-level formatting of internal data streams. It is designed
// to be called concurrently from Python (via pybind11) without the GIL
// serializing the actual arithmetic — every shared structure is guarded
// by its own mutex, and heavy computation is offloaded to a worker pool.
#pragma once

#include <atomic>
#include <condition_variable>
#include <cstdint>
#include <map>
#include <mutex>
#include <queue>
#include <random>
#include <string>
#include <thread>
#include <vector>

namespace zane {

// Inputs Zane weighs when he calculates a probability. All fields are
// expected in [0, 100]; calculate_success_probability() clamps regardless.
struct ProbabilityFactors {
    double danger_level = 0.0;              // Threat assessment of the situation.
    double team_synergy = 0.0;              // How well the Ninja are coordinating.
    double resource_availability = 0.0;     // Equipment / elemental power on hand.
    double historical_success_rate = 0.0;   // Prior outcomes in similar scenarios.
    double complexity_index = 0.0;          // How many moving parts the plan has.
};

// Result of a probability calculation, ready to be quoted verbatim in
// Zane's dialogue (e.g. "I calculate a 73.42% chance of success.").
struct ProbabilityResult {
    uint64_t computation_id = 0;
    double success_probability = 0.0;   // Percentage, 0-100.
    double risk_index = 0.0;            // Percentage, 0-100.
    double confidence_interval_low = 0.0;
    double confidence_interval_high = 0.0;
    std::string analytical_summary;
};

// A raw numerical reading Zane wants formatted into his internal
// diagnostic voice (used both for flavor text and for real telemetry).
struct DataStreamFrame {
    std::string label;
    std::vector<double> values;
    uint64_t timestamp_ns = 0;
};

// Thread-safe engine. One instance is shared across the whole process;
// every public method takes whatever lock it needs internally, so callers
// (including multiple Python asyncio tasks dispatched onto worker threads)
// never need to synchronize externally.
class ZaneAnalytics {
public:
    explicit ZaneAnalytics(unsigned int worker_threads = 2,
                            uint64_t seed = 0 /* 0 => seed from random_device */);
    ~ZaneAnalytics();

    ZaneAnalytics(const ZaneAnalytics&) = delete;
    ZaneAnalytics& operator=(const ZaneAnalytics&) = delete;

    // Synchronous, blocking calculation — computes immediately on the
    // calling thread. Use for latency-sensitive single calculations.
    ProbabilityResult calculate_success_probability(const ProbabilityFactors& factors);

    // Asynchronous submission for when Zane wants to fire off analysis
    // while other work (e.g. the live web search) proceeds in parallel.
    // Returns a computation_id to be polled/awaited.
    uint64_t submit_calculation(const ProbabilityFactors& factors);
    bool try_get_result(uint64_t computation_id, ProbabilityResult& out);
    ProbabilityResult get_result_blocking(uint64_t computation_id,
                                          int timeout_ms = 5000);

    // Renders a DataStreamFrame into Zane's internal-diagnostic voice,
    // e.g. "[ZANE::DATASTREAM] label=WEB_SEARCH_LATENCY n=4 mean=182.3 ...".
    std::string format_data_stream(const DataStreamFrame& frame);

    // Small shared key/value state (e.g. current alert level, mission
    // clock) that both the C++ and Python layers can read and mutate.
    std::map<std::string, double> get_internal_state_snapshot();
    void update_internal_state(const std::string& key, double value);

    // Stops worker threads. Safe to call multiple times; destructor calls
    // it automatically if the caller forgets.
    void shutdown();

private:
    struct Job {
        uint64_t id;
        ProbabilityFactors factors;
    };

    void worker_loop();
    ProbabilityResult compute_locked(const ProbabilityFactors& factors, uint64_t id);

    // --- shared internal state ---
    std::mutex state_mutex_;
    std::map<std::string, double> internal_state_;

    // --- job queue for the worker pool ---
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::queue<Job> job_queue_;

    // --- completed results ---
    std::mutex results_mutex_;
    std::condition_variable results_cv_;
    std::map<uint64_t, ProbabilityResult> results_;

    // --- worker lifecycle ---
    std::vector<std::thread> workers_;
    std::atomic<bool> running_{false};
    std::atomic<uint64_t> next_id_{1};

    // --- RNG, guarded separately since mt19937_64 is not thread-safe ---
    std::mutex rng_mutex_;
    std::mt19937_64 rng_;
};

}  // namespace zane
