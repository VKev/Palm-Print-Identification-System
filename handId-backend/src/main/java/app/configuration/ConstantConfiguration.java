package app.configuration;

import org.springframework.beans.factory.annotation.Value;
import org.springframework.context.annotation.PropertySource;
import org.springframework.context.annotation.PropertySources;
import org.springframework.stereotype.Component;

@Component
@PropertySources({
        @PropertySource("classpath:security.properties"),
        @PropertySource("classpath:other.properties")
})
public class ConstantConfiguration {

    // JWT CONSTANTS
    public final String AUTHENTICATION_HEADER;
    public final String AUTHENTICATION_HEADER_BEARER;
    public final String SECRET_KEY;
    public final Long JWT_TOKEN_EXPIRATION;
    public final Long REFRESH_TOKEN_EXPIRATION;
    public final Boolean JWT_EXPIRED_ENABLE;
    public final Boolean JWT_EXPIRED_DISABLE;
    public final Boolean JWT_REVOKED_ENABLE;
    public final Boolean JWT_REVOKED_DISABLE;

    // API CONSTANTS
    public final String API_ALL_VIDEO;
    public final String API_ALL_AUTH;
    public final String LOGOUT_HANDLER_URL;

    // UI URL
    public final String UI_BASE_URL;

    // CORS
    public final String CORS_ALLOWED_HEADER;
    public final Long CORS_MAX_AGE;
    public final String CORS_PATTERN;

    // LOCAL FOLDER PATH
    public final String LOCAL_VIDEO_PATH;
    public final String LOCAL_IMAGES_PATH;
    public final String IMAGES_EXTENSION_TYPE;
    public final Integer EXTRACTOR_IMAGE_FRAME_JUMP;

    public ConstantConfiguration(
        @Value("${auth.header}") String authenticationHeader,
        @Value("${auth.header.bearer}") String authenticationHeaderBearer,
        @Value("${jwt.secret-key}") String jwtSecretKey,
        @Value("${jwt.expiration}") Long jwtExpiration,
        @Value("${jwt.refresh-token.expiration}") Long refreshTokenExpiration,
        @Value("${jwt.revoked.disable}") Boolean revokeDisable,
        @Value("${jwt.expired.disable}")Boolean expiredDisable,
        @Value("${jwt.revoked.enable}") Boolean revokeEnable,
        @Value("${jwt.expired.enable}") Boolean expiredEnable,
        @Value("${api.url.all.auth}") String apiAllAuth,
        @Value("${url.logout}") String apiLogoutHandler,
        @Value("${url.ui}") String uiBaseUrl,
        @Value("${url.cors.allowed-header}") String corsAllowedHeader,
        @Value("${url.cors.age}") Long maxAge,
        @Value("${url.cors.pattern}") String corsPattern,
        @Value("${local.folder.path.video}") String folderPathVideo,
        @Value("${local.folder.path.images}") String folderPathImages,
        @Value("${local.image.type}") String imageExtensionType,
        @Value("${extractor.image.frame-jump}") Integer imageFrameJump,
        @Value("${api.url.all.video}") String apiAllVideo
    ) {
        this.AUTHENTICATION_HEADER = authenticationHeader;
        this.AUTHENTICATION_HEADER_BEARER = authenticationHeaderBearer;
        this.SECRET_KEY = jwtSecretKey;
        this.JWT_TOKEN_EXPIRATION = jwtExpiration;
        this.REFRESH_TOKEN_EXPIRATION = refreshTokenExpiration;
        this.JWT_REVOKED_DISABLE = revokeDisable;
        this.JWT_EXPIRED_DISABLE = expiredDisable;
        this.JWT_EXPIRED_ENABLE = expiredEnable;
        this.JWT_REVOKED_ENABLE = revokeEnable;
        this.API_ALL_AUTH = apiAllAuth;
        this.LOGOUT_HANDLER_URL = apiLogoutHandler;
        this.UI_BASE_URL = uiBaseUrl;
        this.CORS_ALLOWED_HEADER = corsAllowedHeader;
        this.CORS_MAX_AGE = maxAge;
        this.CORS_PATTERN = corsPattern;
        this.LOCAL_VIDEO_PATH = folderPathVideo;
        this.LOCAL_IMAGES_PATH = folderPathImages;
        this.IMAGES_EXTENSION_TYPE = imageExtensionType;
        this.EXTRACTOR_IMAGE_FRAME_JUMP = imageFrameJump;
        this.API_ALL_VIDEO = apiAllVideo;
    }

}
