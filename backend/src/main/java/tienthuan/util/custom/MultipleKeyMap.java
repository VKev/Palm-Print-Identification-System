package tienthuan.util.custom;

import lombok.AllArgsConstructor;
import lombok.Getter;
import lombok.Setter;
import java.util.ArrayList;
import java.util.HashMap;
import java.util.List;
import java.util.Map;

@Getter
@Setter
@AllArgsConstructor
public class MultipleKeyMap<K1, K2, V> {

    private Map<K1, Map<K2, V>> internalMap;

    public MultipleKeyMap() {
        this.internalMap = new HashMap<>();
    }

    /**
     * Put a value into the map with two keys
     * @param key1 First key
     * @param key2 Second key
     * @param value Value to store
     * @return Previous value if existed, null otherwise
     */
    public V put(K1 key1, K2 key2, V value) {
        Map<K2, V> innerMap = internalMap.computeIfAbsent(key1, k -> new HashMap<>());
        return innerMap.put(key2, value);
    }

    /**
     * Get a value from the map using both keys
     * @param key1 First key
     * @param key2 Second key
     * @return Value if found, null otherwise
     */
    public V get(K1 key1, K2 key2) {
        Map<K2, V> innerMap = internalMap.get(key1);
        return innerMap != null ? innerMap.get(key2) : null;
    }

    /**
     * Remove a value from the map using both keys
     * @param key1 First key
     * @param key2 Second key
     * @return Removed value if existed, null otherwise
     */
    public V remove(K1 key1, K2 key2) {
        Map<K2, V> innerMap = internalMap.get(key1);
        if (innerMap != null) {
            V removedValue = innerMap.remove(key2);
            if (innerMap.isEmpty()) {
                internalMap.remove(key1);
            }
            return removedValue;
        }
        return null;
    }

    /**
     * Get all values associated with the first key as a List
     * @param key1 First key
     * @return List of values associated with key1, empty list if key1 doesn't exist
     */
    public List<V> getValuesByKey1(K1 key1) {
        Map<K2, V> innerMap = internalMap.get(key1);
        if (innerMap != null) {
            return new ArrayList<>(innerMap.values());
        }
        return new ArrayList<>();
    }

    /**
     * Check if the map contains both specified keys
     * @param key1 First key
     * @param key2 Second key
     * @return true if both keys exist, false otherwise
     */
    public boolean containsKeys(K1 key1, K2 key2) {
        Map<K2, V> innerMap = internalMap.get(key1);
        return innerMap != null && innerMap.containsKey(key2);
    }

    /**
     * Check if the map contains the specified first key
     * @param key1 First key to check
     * @return true if key1 exists, false otherwise
     */
    public boolean containsKey1(K1 key1) {
        return internalMap.containsKey(key1);
    }

    /**
     * Check if the map contains the specified second key in any of its inner maps
     * @param key2 Second key to check
     * @return true if key2 exists in any inner map, false otherwise
     */
    public boolean containsKey2(K2 key2) {
        return internalMap.values().stream()
                .anyMatch(innerMap -> innerMap.containsKey(key2));
    }

    /**
     * Get all values associated with the first key
     * @param key1 First key
     * @return Map of second keys to values, null if key1 doesn't exist
     */
    public Map<K2, V> getAllForKey1(K1 key1) {
        return internalMap.get(key1);
    }

    /**
     * Clear all entries from the map
     */
    public void clear() {
        internalMap.clear();
    }

    /**
     * Get the size of the map (total number of values stored)
     * @return Total number of values in the map
     */
    public int size() {
        return internalMap.values().stream()
                .mapToInt(Map::size)
                .sum();
    }

    /**
     * Check if the map is empty
     * @return true if no entries exist, false otherwise
     */
    public boolean isEmpty() {
        return internalMap.isEmpty();
    }
}
