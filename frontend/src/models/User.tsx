
export type UserProfile = {
    id: number;
    username: string;
    fullname: string;
    role: string;
}

export type UserProfileToken = {
    id: number;
    username: string;
    token: string;
}

export const Role = {
    ADMIN : 'ADMIN',
    STAFF : 'STAFF'
}

export type AuthTokens = {
    access_token: string,
    refresh_token: string
}

export type Account = {
    id: number,
    username: string,
    fullname: string,
    role: string,
    isEnable: boolean
}