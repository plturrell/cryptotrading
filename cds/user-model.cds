namespace com.rex.cryptotrading.user;

using { managed, cuid } from '@sap/cds/common';

/**
 * Users Entity
 * Core user management
 */
entity Users : managed {
    key id          : Integer @title: 'ID';
    username        : String(50) @title: 'Username' not null;
    email           : String(100) @title: 'Email' not null;
    passwordHash    : String(255) @title: 'Password Hash' not null;
    firstName       : String(50) @title: 'First Name' not null;
    lastName        : String(50) @title: 'Last Name' not null;
    role            : String(20) @title: 'Role' not null;
    status          : String(20) @title: 'Status' not null default 'active';
    avatarUrl       : String(255) @title: 'Avatar URL';
    phone           : String(20) @title: 'Phone Number';
    language        : String(5) @title: 'Language' default 'en';
    timezone        : String(50) @title: 'Timezone' default 'UTC';
    twoFactorEnabled: Boolean @title: '2FA Enabled' default false;
    twoFactorSecret : String(100) @title: '2FA Secret';
    apiKey          : String(100) @title: 'API Key';
    lastLogin       : Timestamp @title: 'Last Login';
    loginAttempts   : Integer @title: 'Login Attempts' default 0;
    lockedUntil     : Timestamp @title: 'Locked Until';
    isActive        : Boolean @title: 'Is Active' default true;
    
    // Navigation
    sessions        : Composition of many UserSessions on sessions.user = $self;
    preferences     : Composition of one UserPreferences on preferences.user = $self;
    auditLog        : Composition of many UserAuditLog on auditLog.user = $self;
    apiCredentials  : Composition of many APICredentials on apiCredentials.user = $self;
}

/**
 * User Preferences Entity
 * User-specific settings and preferences
 */
entity UserPreferences : managed {
    key id          : Integer @title: 'ID';
    user            : Association to Users @title: 'User';
    theme           : String(20) @title: 'Theme' default 'dark';
    language        : String(5) @title: 'Language' default 'en';
    currency        : String(10) @title: 'Currency' default 'USD';
    notificationsEnabled: Boolean @title: 'Notifications Enabled' default true;
    emailNotifications: Boolean @title: 'Email Notifications' default true;
    smsNotifications: Boolean @title: 'SMS Notifications' default false;
    tradingView     : String(20) @title: 'Trading View' default 'advanced';
    defaultExchange : String(50) @title: 'Default Exchange' default 'binance';
    chartIndicators : LargeString @title: 'Chart Indicators (JSON)';
    favoritePairs   : LargeString @title: 'Favorite Pairs (JSON)';
    customSettings  : LargeString @title: 'Custom Settings (JSON)';
}

/**
 * User Sessions Entity
 * Active user sessions management
 */
entity UserSessions : managed {
    key id          : Integer @title: 'ID';
    user            : Association to Users @title: 'User';
    sessionToken    : String(255) @title: 'Session Token' not null;
    ipAddress       : String(45) @title: 'IP Address';
    userAgent       : String(500) @title: 'User Agent';
    deviceInfo      : LargeString @title: 'Device Info';
    location        : String(100) @title: 'Location';
    expiresAt       : Timestamp @title: 'Expires At' not null;
    lastActivity    : Timestamp @title: 'Last Activity';
    isActive        : Boolean @title: 'Is Active' default true;
}

/**
 * User Audit Log Entity
 * Track all user actions for security
 */
entity UserAuditLog : managed {
    key id          : Integer @title: 'ID';
    user            : Association to Users @title: 'User';
    action          : String(50) @title: 'Action' not null;
    details         : LargeString @title: 'Details';
    ipAddress       : String(45) @title: 'IP Address';
    userAgent       : String(500) @title: 'User Agent';
    status          : String(20) @title: 'Status';
    errorMessage    : String(500) @title: 'Error Message';
}

/**
 * Conversation Sessions Entity
 * User conversation sessions
 */
entity ConversationSessions : cuid, managed {
    user            : Association to Users @title: 'User';
    sessionName     : String(100) @title: 'Session Name';
    sessionType     : String(50) @title: 'Session Type';
    status          : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        PAUSED;
        COMPLETED;
        EXPIRED;
    } default 'ACTIVE';
    startedAt       : Timestamp @title: 'Started At';
    endedAt         : Timestamp @title: 'Ended At';
    metadata        : LargeString @title: 'Session Metadata (JSON)';
    
    // Navigation
    messages        : Composition of many ConversationMessages on messages.session = $self;
    history         : Composition of many ConversationHistory on history.session = $self;
}

/**
 * Conversation Messages Entity
 * Individual messages in conversations
 */
entity ConversationMessages : cuid, managed {
    session         : Association to ConversationSessions @title: 'Session';
    messageType     : String(20) @title: 'Message Type' @assert.range enum {
        USER;
        ASSISTANT;
        SYSTEM;
        ERROR;
    };
    content         : LargeString @title: 'Message Content';
    role            : String(20) @title: 'Role';
    metadata        : LargeString @title: 'Metadata (JSON)';
    timestamp       : Timestamp @title: 'Timestamp';
    tokens          : Integer @title: 'Token Count';
}

/**
 * Conversation History Entity
 * Historical conversation data
 */
entity ConversationHistory : cuid, managed {
    session         : Association to ConversationSessions @title: 'Session';
    summary         : LargeString @title: 'Conversation Summary';
    keyPoints       : LargeString @title: 'Key Points (JSON)';
    decisions       : LargeString @title: 'Decisions Made (JSON)';
    followUps       : LargeString @title: 'Follow-ups (JSON)';
    sentiment       : String(20) @title: 'Overall Sentiment';
    quality         : Decimal(3,2) @title: 'Quality Score';
}

/**
 * API Credentials Entity
 * API access credentials for users
 */
entity APICredentials : cuid, managed {
    user            : Association to Users @title: 'User';
    apiKey          : String(100) @title: 'API Key';
    apiSecret       : String(200) @title: 'API Secret (Hashed)';
    name            : String(50) @title: 'Key Name';
    description     : String(200) @title: 'Description';
    permissions     : LargeString @title: 'Permissions (JSON)';
    status          : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        REVOKED;
        EXPIRED;
    } default 'ACTIVE';
    expiresAt       : Timestamp @title: 'Expires At';
    lastUsed        : Timestamp @title: 'Last Used';
    usageCount      : Integer @title: 'Usage Count' default 0;
    rateLimit       : Integer @title: 'Rate Limit (req/hour)' default 1000;
}

// Analytics Views
view ActiveUsers as select from Users {
    *
} where isActive = true;

view RecentSessions as select from ConversationSessions {
    *
} where createdAt >= $now - 604800000; // Last 7 days

view UserActivity as select from Users {
    id,
    username,
    email,
    isActive,
    count(sessions.id) as totalSessions : Integer,
    count(apiCredentials.id) as totalApiKeys : Integer
} group by id, username, email, isActive;