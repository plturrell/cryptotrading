const cds = require('@sap/cds');
const express = require('express');
const cors = require('cors');
const helmet = require('helmet');
const compression = require('compression');
const rateLimit = require('express-rate-limit');
const winston = require('winston');

// Initialize logger
const logger = winston.createLogger({
    level: 'info',
    format: winston.format.combine(
        winston.format.timestamp(),
        winston.format.errors({ stack: true }),
        winston.format.json()
    ),
    transports: [
        new winston.transports.File({ filename: 'logs/error.log', level: 'error' }),
        new winston.transports.File({ filename: 'logs/combined.log' }),
        new winston.transports.Console({
            format: winston.format.simple()
        })
    ]
});

// Rate limiting
const limiter = rateLimit({
    windowMs: 15 * 60 * 1000, // 15 minutes
    max: 1000, // limit each IP to 1000 requests per windowMs
    message: 'Too many requests from this IP, please try again later.'
});

module.exports = cds.server;

cds.on('bootstrap', (app) => {
    // Security middleware
    app.use(helmet({
        contentSecurityPolicy: {
            directives: {
                defaultSrc: ["'self'"],
                scriptSrc: ["'self'", "'unsafe-inline'", "'unsafe-eval'"],
                styleSrc: ["'self'", "'unsafe-inline'"],
                imgSrc: ["'self'", "data:", "https:"],
                connectSrc: ["'self'", "ws:", "wss:"]
            }
        }
    }));
    
    app.use(compression());
    app.use(cors({
        origin: process.env.ALLOWED_ORIGINS?.split(',') || ['http://localhost:8080'],
        credentials: true
    }));
    app.use(limiter);
    
    // Health check endpoint
    app.get('/health', (req, res) => {
        res.status(200).json({
            status: 'healthy',
            timestamp: new Date().toISOString(),
            uptime: process.uptime(),
            version: require('../package.json').version
        });
    });
    
    // API documentation endpoint
    app.get('/api-docs', (req, res) => {
        res.json({
            message: 'Crypto Trading Platform API',
            version: '1.0.0',
            endpoints: {
                health: '/health',
                marketData: '/api/market-data',
                trading: '/api/trading',
                portfolio: '/api/portfolio',
                analytics: '/api/analytics',
                risk: '/api/risk'
            }
        });
    });
    
    logger.info('Crypto Trading Platform server initialized');
});

cds.on('listening', ({ server, url }) => {
    logger.info(`Crypto Trading Platform server listening at ${url}`);
    logger.info('Available CDS services:');
    
    // Get actual registered services
    const services = cds.services;
    Object.keys(services).forEach(serviceName => {
        const service = services[serviceName];
        if (service.path && service.path !== '/') {
            logger.info(`- ${serviceName} → ${service.path}`);
        }
    });
});

// Error handling
process.on('unhandledRejection', (reason, promise) => {
    logger.error('Unhandled Rejection at:', promise, 'reason:', reason);
});

process.on('uncaughtException', (error) => {
    logger.error('Uncaught Exception:', error);
    process.exit(1);
});
