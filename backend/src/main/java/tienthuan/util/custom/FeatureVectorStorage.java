package tienthuan.util.custom;

import java.util.HashMap;
import java.util.List;

public class FeatureVectorStorage extends HashMap<String, List<FeatureVector>> {

    public FeatureVectorStorage() {
        super();
    }

    public boolean addVector(String key, FeatureVector featureVector) throws Exception {
        if (this.containsKey(key)) {
            return this.get(key).add(featureVector);
        }
        else {
            this.put(key, List.of(featureVector));
            return true;
        }
    }

}
