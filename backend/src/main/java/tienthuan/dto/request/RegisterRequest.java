package tienthuan.dto.request;

public record RegisterRequest (
        String username,
        String fullname,
        String password
){ }
