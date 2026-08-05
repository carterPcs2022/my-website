#include "zane/analytics.hpp"

#include <algorithm>
#include <chrono>
#include <cmath>
#include <numeric>
#include <sstream>
#include <stdexcept>

namespace zane {

namespace {

double clamp_pct(double v) {
    return std::clamp(v, 0.0, 100.0);
}

uint64_t now_ns() {
    return static_cast<uint64_t>(
        std::chrono::duration_cast<std::chrono::nanoseconds>(
            std::chrono::system_clock::now().time_since_epoch())
            .count());
}

}  // namespace

ZaneAnalytics::ZaneAnalytics(unsigned int worker_threads, uint64_t seed) {
    if (worker_threads == 0) {
        worker_threads = 1;
    }

    if (seed == 0) {
        std::random_device rd;
        // Mix several random_device draws into the seed sequence so
        // platforms with low-entropy random_device implementations still
        // get well-distributed 64-bit seeds.
        std::seed_seq seq{rd(), rd(), rd(), rd()};
        std::mt19937_64 seeded(seq);
        rng_ = seeded;
    } else {
        rng_ = std::mt19937_64(seed);
    }

    running_.store(true);
    workers_.reserve(worker_threads);
    for (unsigned int i = 0; i < worker_threads; ++i) {
        workers_.emplace_back(&ZaneAnalytics::worker_loop, this);
    }

    // Reasonable defaults so Python can read state before it ever writes.
    internal_state_["alert_level"] = 0.0;
    internal_state_["mission_clock_s"] = 0.0;
    internal_state_["nindroid_core_temp_c"] = 36.6;
}

ZaneAnalytics::~ZaneAnalytics() {
    shutdown();
}

void ZaneAnalytics::shutdown() {
    bool was_running = running_.exchange(false);
    if (!was_running) {
        return;
    }
    queue_cv_.notify_all();
    for (auto& t : workers_) {
        if (t.joinable()) {
            t.join();
        }
    }
    workers_.clear();
}

void ZaneAnalytics::worker_loop() {
    while (true) {
        Job job;
        {
            std::unique_lock<std::mutex> lock(queue_mutex_);
            queue_cv_.wait(lock, [this] { return !job_queue_.empty() || !running_.load(); });
            if (!running_.load() && job_queue_.empty()) {
                return;
            }
            job = job_queue_.front();
            job_queue_.pop();
        }

        ProbabilityResult result = compute_locked(job.factors, job.id);

        {
            std::lock_guard<std::mutex> lock(results_mutex_);
            results_[job.id] = result;
        }
        results_cv_.notify_all();
    }
}

ProbabilityResult ZaneAnalytics::compute_locked(const ProbabilityFactors& raw, uint64_t id) {
    ProbabilityFactors f;
    f.danger_level = clamp_pct(raw.danger_level);
    f.team_synergy = clamp_pct(raw.team_synergy);
    f.resource_availability = clamp_pct(raw.resource_availability);
    f.historical_success_rate = clamp_pct(raw.historical_success_rate);
    f.complexity_index = clamp_pct(raw.complexity_index);

    // Weighted heuristic model. Weights sum to 1.0 across five factors;
    // danger and complexity contribute their *inverse* since higher
    // danger/complexity should push success probability down.
    const double base =
        f.team_synergy * 0.28 +
        f.resource_availability * 0.22 +
        f.historical_success_rate * 0.30 +
        (100.0 - f.danger_level) * 0.12 +
        (100.0 - f.complexity_index) * 0.08;

    // Stochastic variance correction: real Nindroid sensors have noise.
    // Standard deviation scales up with complexity — more moving parts,
    // less predictable outcome — bounded to keep the readout believable.
    const double sigma = 1.5 + (f.complexity_index / 100.0) * 4.0;

    double jitter;
    {
        std::lock_guard<std::mutex> lock(rng_mutex_);
        std::normal_distribution<double> dist(0.0, sigma);
        jitter = dist(rng_);
    }

    const double success = clamp_pct(base + jitter);
    const double risk = clamp_pct(100.0 - success + (f.danger_level * 0.05));

    // 90%-style confidence band derived from sigma, clamped to [0, 100].
    const double half_width = 1.645 * sigma;
    const double ci_low = clamp_pct(success - half_width);
    const double ci_high = clamp_pct(success + half_width);

    std::ostringstream summary;
    summary.setf(std::ios::fixed);
    summary.precision(2);
    summary << "Weighted nindroid heuristic model (team_synergy=" << f.team_synergy
            << ", resources=" << f.resource_availability
            << ", history=" << f.historical_success_rate
            << ", danger=" << f.danger_level
            << ", complexity=" << f.complexity_index
            << ") with stochastic variance correction (sigma=" << sigma << ").";

    ProbabilityResult result;
    result.computation_id = id;
    result.success_probability = success;
    result.risk_index = risk;
    result.confidence_interval_low = ci_low;
    result.confidence_interval_high = ci_high;
    result.analytical_summary = summary.str();
    return result;
}

ProbabilityResult ZaneAnalytics::calculate_success_probability(const ProbabilityFactors& factors) {
    const uint64_t id = next_id_.fetch_add(1);
    return compute_locked(factors, id);
}

uint64_t ZaneAnalytics::submit_calculation(const ProbabilityFactors& factors) {
    const uint64_t id = next_id_.fetch_add(1);
    {
        std::lock_guard<std::mutex> lock(queue_mutex_);
        job_queue_.push(Job{id, factors});
    }
    queue_cv_.notify_one();
    return id;
}

bool ZaneAnalytics::try_get_result(uint64_t computation_id, ProbabilityResult& out) {
    std::lock_guard<std::mutex> lock(results_mutex_);
    auto it = results_.find(computation_id);
    if (it == results_.end()) {
        return false;
    }
    out = it->second;
    results_.erase(it);
    return true;
}

ProbabilityResult ZaneAnalytics::get_result_blocking(uint64_t computation_id, int timeout_ms) {
    std::unique_lock<std::mutex> lock(results_mutex_);
    const bool ready = results_cv_.wait_for(
        lock, std::chrono::milliseconds(timeout_ms),
        [this, computation_id] { return results_.find(computation_id) != results_.end(); });

    if (!ready) {
        throw std::runtime_error("ZaneAnalytics: computation " +
                                  std::to_string(computation_id) + " timed out");
    }

    auto it = results_.find(computation_id);
    ProbabilityResult result = it->second;
    results_.erase(it);
    return result;
}

std::string ZaneAnalytics::format_data_stream(const DataStreamFrame& frame) {
    const auto& values = frame.values;
    const size_t n = values.size();

    double mean = 0.0, min_v = 0.0, max_v = 0.0, stddev = 0.0;
    if (n > 0) {
        mean = std::accumulate(values.begin(), values.end(), 0.0) / static_cast<double>(n);
        min_v = *std::min_element(values.begin(), values.end());
        max_v = *std::max_element(values.begin(), values.end());

        double sq_sum = 0.0;
        for (double v : values) {
            sq_sum += (v - mean) * (v - mean);
        }
        stddev = std::sqrt(sq_sum / static_cast<double>(n));
    }

    const uint64_t ts = frame.timestamp_ns != 0 ? frame.timestamp_ns : now_ns();

    std::ostringstream out;
    out.setf(std::ios::fixed);
    out.precision(3);
    out << "[ZANE::DATASTREAM] label=" << frame.label
        << " n=" << n
        << " mean=" << mean
        << " min=" << min_v
        << " max=" << max_v
        << " stddev=" << stddev
        << " ts_ns=" << ts;
    return out.str();
}

std::map<std::string, double> ZaneAnalytics::get_internal_state_snapshot() {
    std::lock_guard<std::mutex> lock(state_mutex_);
    return internal_state_;  // copy out under the lock, safe to use lock-free after
}

void ZaneAnalytics::update_internal_state(const std::string& key, double value) {
    std::lock_guard<std::mutex> lock(state_mutex_);
    internal_state_[key] = value;
}

}  // namespace zane
