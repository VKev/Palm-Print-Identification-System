package tienthuan.util;

import tienthuan.util.custom.MultipleKeyMap;
import java.util.List;

public class FeatureVectorUtil {

    public static final Integer FRAMES_LIMITATION = 30;
    public static final MultipleKeyMap<String, Integer, List<Double>> featureVectorStorage = new MultipleKeyMap<>();

    public synchronized static void storeFeatureVector(String uuid, List<Double> featureVector) {
        if (!isExistUUID(uuid)) {
            featureVectorStorage.put(uuid, 1, featureVector);
        } else {
            Integer key2 = generateKey2(uuid);
            featureVectorStorage.put(uuid, key2, featureVector);
        }
    }

    public static List<List<Double>> getAllVectors(String uuid) {
        if (!isExistUUID(uuid)) {
            return null;
        }
        return featureVectorStorage.getValuesByKey1(uuid);
    }

    public static int getNumberOfVectors(String uuid) {
        if (!isExistUUID(uuid)) {
            return 0;
        }
        return featureVectorStorage.getAllForKey1(uuid).size();
    }

    private synchronized static boolean isExistUUID(String uuid) {
        return featureVectorStorage.containsKey1(uuid);
    }

    private synchronized static Integer generateKey2(String uuid) {
        if (!isExistUUID(uuid)) {
            return 1;
        }
        return featureVectorStorage.getAllForKey1(uuid).size() + 1;
    }

}
