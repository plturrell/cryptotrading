/**
 * Enhanced A2A Features for 100% Protocol Conformance
 * Adds: Large File Transfer, Chunking, Compression, Encryption, Transactions
 */

const zlib = require("zlib");
const crypto = require("crypto");
const AWS = require("aws-sdk");
const { v4: uuidv4 } = require("uuid");

// Initialize AWS S3 client
const s3 = new AWS.S3({
    region: process.env.AWS_REGION || "us-east-1",
    accessKeyId: process.env.AWS_ACCESS_KEY_ID,
    secretAccessKey: process.env.AWS_SECRET_ACCESS_KEY
});

/**
 * Enhanced A2A Feature Set
 * Add these methods to your A2AService class
 */
class A2AEnhancedFeatures {

    // ============== PHASE 1: LARGE FILE TRANSFER ==============

    async sendLargeFile(fromAgentId, toAgentId, fileName, fileData, contentType) {
        try {
            const s3Bucket = process.env.A2A_S3_BUCKET || "cryptotrading-a2a-files";
            const s3Key = `a2a-transfers/${fromAgentId}/${Date.now()}-${fileName}`;

            // Upload to S3
            const uploadParams = {
                Bucket: s3Bucket,
                Key: s3Key,
                Body: Buffer.from(fileData, "base64"),
                ContentType: contentType || "application/octet-stream",
                Metadata: {
                    fromAgent: fromAgentId,
                    toAgent: toAgentId,
                    timestamp: new Date().toISOString()
                }
            };

            const uploadResult = await s3.upload(uploadParams).promise();

            // Create A2A message with S3 reference
            const message = {
                messageType: "FILE_TRANSFER",
                payload: JSON.stringify({
                    s3Url: uploadResult.Location,
                    s3Key: s3Key,
                    fileName: fileName,
                    fileSize: Buffer.byteLength(fileData, "base64"),
                    contentType: contentType,
                    checksum: crypto.createHash("sha256").update(fileData).digest("hex")
                })
            };

            return {
                success: true,
                s3Url: uploadResult.Location,
                message: message
            };

        } catch (error) {
            // Large file transfer failed: error
            return { success: false, error: error.message };
        }
    }

    async downloadLargeFile(s3Key) {
        try {
            const s3Bucket = process.env.A2A_S3_BUCKET || "cryptotrading-a2a-files";

            const params = {
                Bucket: s3Bucket,
                Key: s3Key
            };

            const data = await s3.getObject(params).promise();

            return {
                success: true,
                data: data.Body,
                contentType: data.ContentType,
                metadata: data.Metadata
            };

        } catch (error) {
            // File download failed: error
            return { success: false, error: error.message };
        }
    }

    // ============== PHASE 2: MESSAGE CHUNKING ==============

    async chunkMessage(payload, chunkSize = 65536) {
        const correlationId = uuidv4();
        const payloadString = typeof payload === "string"
            ? payload
            : JSON.stringify(payload);

        const chunks = [];
        for (let i = 0; i < payloadString.length; i += chunkSize) {
            chunks.push({
                correlationId: correlationId,
                chunkIndex: Math.floor(i / chunkSize),
                data: payloadString.slice(i, i + chunkSize)
            });
        }

        return {
            correlationId: correlationId,
            totalChunks: chunks.length,
            chunks: chunks
        };
    }

    reassembleChunks(chunks) {
        // Sort chunks by index
        chunks.sort((a, b) => a.chunkIndex - b.chunkIndex);

        // Concatenate data
        const reassembled = chunks.map(c => c.data).join("");

        try {
            return JSON.parse(reassembled);
        } catch {
            return reassembled;
        }
    }

    // ============== PHASE 3: COMPRESSION ==============

    compressPayload(payload, algorithm = "gzip") {
        const payloadString = typeof payload === "string"
            ? payload
            : JSON.stringify(payload);

        if (payloadString.length < 10240) {
            // Don't compress small payloads
            return {
                compressed: false,
                data: payloadString,
                originalSize: payloadString.length
            };
        }

        let compressed;
        switch (algorithm) {
        case "gzip":
            compressed = zlib.gzipSync(payloadString);
            break;
        case "deflate":
            compressed = zlib.deflateSync(payloadString);
            break;
        case "brotli":
            compressed = zlib.brotliCompressSync(payloadString);
            break;
        default:
            compressed = zlib.gzipSync(payloadString);
        }

        return {
            compressed: true,
            algorithm: algorithm,
            data: compressed.toString("base64"),
            originalSize: payloadString.length,
            compressedSize: compressed.length,
            compressionRatio: (1 - compressed.length / payloadString.length) * 100
        };
    }

    decompressPayload(compressedData, algorithm = "gzip") {
        if (!compressedData.compressed) {
            return compressedData.data;
        }

        const buffer = Buffer.from(compressedData.data, "base64");
        let decompressed;

        switch (algorithm) {
        case "gzip":
            decompressed = zlib.gunzipSync(buffer);
            break;
        case "deflate":
            decompressed = zlib.inflateSync(buffer);
            break;
        case "brotli":
            decompressed = zlib.brotliDecompressSync(buffer);
            break;
        default:
            decompressed = zlib.gunzipSync(buffer);
        }

        const result = decompressed.toString("utf8");

        try {
            return JSON.parse(result);
        } catch {
            return result;
        }
    }

    // ============== PHASE 4: ENCRYPTION ==============

    generateKeyPair() {
        const { publicKey, privateKey } = crypto.generateKeyPairSync("rsa", {
            modulusLength: 2048,
            publicKeyEncoding: {
                type: "spki",
                format: "pem"
            },
            privateKeyEncoding: {
                type: "pkcs8",
                format: "pem"
            }
        });

        return { publicKey, privateKey };
    }

    encryptPayload(payload, recipientPublicKey) {
        const payloadString = typeof payload === "string"
            ? payload
            : JSON.stringify(payload);

        // Generate AES key for this message
        const aesKey = crypto.randomBytes(32);
        const iv = crypto.randomBytes(16);

        // Encrypt payload with AES
        const cipher = crypto.createCipheriv("aes-256-gcm", aesKey, iv);
        let encrypted = cipher.update(payloadString, "utf8", "hex");
        encrypted += cipher.final("hex");
        const authTag = cipher.getAuthTag();

        // Encrypt AES key with recipient's public key
        const encryptedKey = crypto.publicEncrypt(
            recipientPublicKey,
            aesKey
        ).toString("base64");

        return {
            encrypted: true,
            algorithm: "aes-256-gcm",
            data: encrypted,
            key: encryptedKey,
            iv: iv.toString("base64"),
            authTag: authTag.toString("base64")
        };
    }

    decryptPayload(encryptedData, privateKey) {
        if (!encryptedData.encrypted) {
            return encryptedData.data || encryptedData;
        }

        // Decrypt AES key with private key
        const aesKey = crypto.privateDecrypt(
            privateKey,
            Buffer.from(encryptedData.key, "base64")
        );

        // Decrypt payload with AES
        const decipher = crypto.createDecipheriv(
            "aes-256-gcm",
            aesKey,
            Buffer.from(encryptedData.iv, "base64")
        );

        decipher.setAuthTag(Buffer.from(encryptedData.authTag, "base64"));

        let decrypted = decipher.update(encryptedData.data, "hex", "utf8");
        decrypted += decipher.final("utf8");

        try {
            return JSON.parse(decrypted);
        } catch {
            return decrypted;
        }
    }

    // ============== PHASE 5: DISTRIBUTED TRANSACTIONS ==============

    createTransaction(transactionId, participants) {
        return {
            id: transactionId || uuidv4(),
            participants: participants,
            status: "INITIALIZING",
            preparedAgents: [],
            votes: {},
            startTime: Date.now(),
            operations: []
        };
    }

    async prepareTransaction(transaction, agentId, canCommit) {
        transaction.votes[agentId] = canCommit;

        if (canCommit) {
            transaction.preparedAgents.push(agentId);
        }

        // Check if all participants have voted
        if (Object.keys(transaction.votes).length === transaction.participants.length) {
            // All votes are in
            const allPrepared = transaction.participants.every(
                p => transaction.votes[p] === true
            );

            if (allPrepared) {
                transaction.status = "PREPARED";
                return { action: "COMMIT" };
            } else {
                transaction.status = "ABORTING";
                return { action: "ROLLBACK" };
            }
        }

        return { action: "WAIT" };
    }

    commitTransaction(transaction) {
        transaction.status = "COMMITTED";
        transaction.endTime = Date.now();
        transaction.duration = transaction.endTime - transaction.startTime;

        return {
            success: true,
            transactionId: transaction.id,
            duration: transaction.duration
        };
    }

    rollbackTransaction(transaction, reason) {
        transaction.status = "ROLLED_BACK";
        transaction.endTime = Date.now();
        transaction.rollbackReason = reason;

        return {
            success: false,
            transactionId: transaction.id,
            reason: reason
        };
    }

    // ============== HELPER METHODS ==============

    async processEnhancedMessage(message) {
        const payload = JSON.parse(message.payload);

        // Check for compression
        if (payload.compressed) {
            const decompressed = this.decompressPayload(payload);
            message.payload = JSON.stringify(decompressed);
        }

        // Check for encryption
        if (payload.encrypted) {
            // Recipient would decrypt here using their private key
            // For demo, we'll skip actual decryption
            // Message is encrypted
        }

        // Check for chunking
        if (message.messageType === "CHUNK_START" ||
            message.messageType === "CHUNK_DATA" ||
            message.messageType === "CHUNK_END") {

            // Store chunk and check if complete
            const chunkData = JSON.parse(message.payload);

            if (!this.chunkStore) {
                this.chunkStore = new Map();
            }

            if (!this.chunkStore.has(chunkData.correlationId)) {
                this.chunkStore.set(chunkData.correlationId, []);
            }

            this.chunkStore.get(chunkData.correlationId).push(chunkData);

            if (message.messageType === "CHUNK_END") {
                const chunks = this.chunkStore.get(chunkData.correlationId);
                const reassembled = this.reassembleChunks(chunks);
                this.chunkStore.delete(chunkData.correlationId);

                return {
                    messageType: chunkData.originalType,
                    payload: reassembled
                };
            }

            return null; // Waiting for more chunks
        }

        // Check for file transfer
        if (message.messageType === "FILE_TRANSFER") {
            const fileInfo = JSON.parse(message.payload);
            // File available at: fileInfo.s3Url
            // Recipient could download the file here
        }

        return message;
    }
}

module.exports = A2AEnhancedFeatures;
