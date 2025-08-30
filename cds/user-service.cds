using { com.rex.cryptotrading.user as user } from './user-model';

namespace com.rex.cryptotrading.user.service;

/**
 * User Management Service - RESTful API for User Operations
 */
@path: '/api/odata/v4/UserService'
service UserService {
    
    // Core User Entities
    @odata.draft.enabled
    @cds.redirection.target
    entity Users as projection on user.Users;
    
    entity UserPreferences as projection on user.UserPreferences;
    
    @cds.redirection.target
    entity UserSessions as projection on user.UserSessions;
    
    entity UserAuditLog as projection on user.UserAuditLog;
    
    @cds.redirection.target
    entity ConversationSessions as projection on user.ConversationSessions;
    
    entity ConversationMessages as projection on user.ConversationMessages;
    
    entity ConversationHistory as projection on user.ConversationHistory;
    
    @restrict: [
        { grant: ['READ'], to: 'authenticated-user', where: 'user.ID = $user.id' },
        { grant: '*', to: 'admin' }
    ]
    entity APICredentials as projection on user.APICredentials;
    
    // Analytics Views
    @readonly
    entity ActiveUsers as projection on user.ActiveUsers;
    
    @readonly
    entity RecentSessions as projection on user.RecentSessions;
    
    @readonly
    entity UserActivity as projection on user.UserActivity;
    
    // User Management Actions
    action createUser(
        username: String,
        email: String,
        firstName: String,
        lastName: String,
        role: String
    ) returns {
        userId: String;
        status: String;
        message: String;
    };
    
    action activateUser(userId: String) returns {
        success: Boolean;
        message: String;
    };
    
    action suspendUser(
        userId: String,
        reason: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    action resetPassword(
        userId: String,
        temporaryPassword: String
    ) returns {
        success: Boolean;
        expiresAt: DateTime;
        message: String;
    };
    
    action enable2FA(userId: String) returns {
        success: Boolean;
        qrCode: String;
        backupCodes: array of String;
    };
    
    // API Key Management
    action generateAPIKey(
        userId: String,
        keyName: String,
        permissions: String,
        expiresInDays: Integer
    ) returns {
        apiKey: String;
        apiSecret: String;
        expiresAt: DateTime;
    };
    
    action revokeAPIKey(
        apiKeyId: String,
        reason: String
    ) returns {
        success: Boolean;
        message: String;
    };
    
    // Session Management
    action startConversation(
        userId: String,
        sessionName: String,
        sessionType: String
    ) returns {
        sessionId: String;
        status: String;
    };
    
    action endConversation(
        sessionId: String
    ) returns {
        success: Boolean;
        summary: String;
    };
    
    // User Query Functions
    function getUserProfile(userId: String) returns {
        username: String;
        email: String;
        displayName: String;
        role: String;
        status: String;
        lastLogin: DateTime;
        statistics: {
            totalSessions: Integer;
            totalMessages: Integer;
            avgSessionDuration: Integer;
        };
    };
    
    function getUserActivity(
        userId: String,
        period: String
    ) returns {
        loginCount: Integer;
        sessionCount: Integer;
        messageCount: Integer;
        apiCallCount: Integer;
        activityTimeline: array of {
            timestamp: DateTime;
            action: String;
            details: String;
        };
    };
    
    function getConversationSummary(sessionId: String) returns {
        sessionName: String;
        duration: Integer;
        messageCount: Integer;
        summary: String;
        keyPoints: array of String;
        decisions: array of String;
        sentiment: String;
        qualityScore: Decimal;
    };
    
    function validateAPIKey(apiKey: String) returns {
        isValid: Boolean;
        userId: String;
        permissions: array of String;
        rateLimit: Integer;
        remainingCalls: Integer;
    };
    
    function getUserPermissions(userId: String) returns {
        role: String;
        permissions: array of {
            resource: String;
            actions: array of String;
        };
        restrictions: array of String;
    };
}