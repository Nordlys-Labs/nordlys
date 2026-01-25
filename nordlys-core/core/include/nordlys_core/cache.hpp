#pragma once

#include <filesystem>
#include <list>
#include <memory>
#include <mutex>
#include <shared_mutex>
#include <string>
#include <unordered_map>

namespace nordlys {

  template <typename T> class LruCache {
  public:
    explicit LruCache(size_t max_size = 8) : max_size_(max_size) {}

    std::shared_ptr<T> get(const std::string& path) {
      std::unique_lock lock(mutex_);

      try {
        auto canonical = std::filesystem::canonical(path).string();
        auto current_mtime = std::filesystem::last_write_time(canonical);

        auto it = map_.find(canonical);
        if (it != map_.end()) {
          if (it->second->second.mtime == current_mtime) {
            move_to_front(it->second);
            return it->second->second.value;
          }
          erase(it);
        }
      } catch (const std::filesystem::filesystem_error&) {
        return nullptr;
      }
      return nullptr;
    }

    void put(const std::string& path, std::shared_ptr<T> value) {
      if (max_size_ == 0) return;

      std::unique_lock lock(mutex_);

      std::string canonical;
      std::filesystem::file_time_type current_mtime;
      try {
        canonical = std::filesystem::canonical(path).string();
        current_mtime = std::filesystem::last_write_time(canonical);
      } catch (const std::filesystem::filesystem_error&) {
        return;
      }

      auto it = map_.find(canonical);
      if (it != map_.end()) {
        it->second->second = {value, current_mtime};
        move_to_front(it->second);
        return;
      }

      if (!list_.empty() && list_.size() >= max_size_) {
        auto& back = list_.back();
        map_.erase(back.first);
        list_.pop_back();
      }

      list_.emplace_front(canonical, Entry{value, current_mtime});
      map_[canonical] = list_.begin();
    }

    void clear() {
      std::unique_lock lock(mutex_);
      list_.clear();
      map_.clear();
    }

    size_t size() const {
      std::shared_lock lock(mutex_);
      return list_.size();
    }

  private:
    struct Entry {
      std::shared_ptr<T> value;
      std::filesystem::file_time_type mtime;
    };

    using ListType = std::list<std::pair<std::string, Entry>>;
    using ListIterator = typename ListType::iterator;

    void move_to_front(ListIterator it) { list_.splice(list_.begin(), list_, it); }

    void erase(typename std::unordered_map<std::string, ListIterator>::iterator map_it) {
      list_.erase(map_it->second);
      map_.erase(map_it);
    }

    mutable std::shared_mutex mutex_;
    ListType list_;
    std::unordered_map<std::string, ListIterator> map_;
    size_t max_size_;
  };

}  // namespace nordlys
