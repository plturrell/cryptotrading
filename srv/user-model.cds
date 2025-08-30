namespace com.rex.cryptotrading.user;

using { managed, cuid } from '@sap/cds/common';

/**
 * Users Entity
 * Core user management
 */
entity Users : cuid, managed {
    username        : String(50) @title: 'Username' @mandatory;
    email           : String(100) @title: 'Email' @mandatory;
    firstName       : String(50) @title: 'First Name';
    lastName        : String(50) @title: 'Last Name';
    displayName     : String(100) @title: 'Display Name';
    status          : String(20) @title: 'Status' @assert.range enum {
        ACTIVE;
        INACTIVE;
        SUSPENDED;
        PENDING;
    } default 'PENDING';
    role            : String(30) @title: 'Role' @assert.range enum {
        ADMIN;
        TRADER;
        ANALYST;
        VIEWER;
        API_USER;
    } default 'VIEWER';
    lastLogin       : Timestamp @title: 'Last Login';
    loginCount      : Integer @title: 'Login Count' default 0;
    preferences     : LargeString @title: 'User Preferences (JSON)';
    twoFactorEnabled: Boolean @title: '2FA Enabled' default false;
    apiKeysEnabled  : Boolean @title: 'API Keys Enabled' default false;
    
    // Navigation
    sessions        : Composition of many ConversationSessions on sessions.user = $self;
    apiCredentials  : Composition of many APICredentials on apiCredentials.user = $self;
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
} where status = 'ACTIVE';

view RecentSessions as select from ConversationSessions {
    *
} where createdAt >= $now - 604800000; // Last 7 days

view UserActivity as select from Users {
    ID,
    username,
    email,
    role,
    lastLogin,
    loginCount,
    count(sessions.ID) as totalSessions : Integer,
    count(apiCredentials.ID) as totalApiKeys : Integer
} group by ID, username, email, role, lastLogin, loginCount;