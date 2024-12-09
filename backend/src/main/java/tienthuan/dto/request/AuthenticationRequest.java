package tienthuan.dto.request;

public record AuthenticationRequest (
        String username,
        String password
) { }
