"""
Distributed Coordination System for Strands Framework
Implements distributed algorithms for coordination, consensus, and synchronization
"""
import asyncio
import hashlib
import json
import logging
import time
import uuid
from collections import defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timedelta
from enum import Enum
from typing import Any, Callable, Dict, List, Optional, Set, Tuple


class CoordinationState(Enum):
    """States for distributed coordination"""

    ACTIVE = "active"
    ELECTING = "electing"
    FOLLOWING = "following"
    FAILED = "failed"


class LockState(Enum):
    """Distributed lock states"""

    AVAILABLE = "available"
    HELD = "held"
    CONTENDED = "contended"
    RELEASED = "released"


class TransactionPhase(Enum):
    """Two-Phase Commit phases"""

    PREPARE = "prepare"
    COMMIT = "commit"
    ABORT = "abort"


@dataclass
class DistributedLock:
    """Distributed lock structure"""

    resource_id: str
    holder_id: Optional[str]
    state: LockState
    acquired_at: Optional[datetime]
    expires_at: Optional[datetime]
    waiters: List[str] = field(default_factory=list)
    version: int = 0


@dataclass
class LeaderElection:
    """Leader election data"""

    term: int
    leader_id: Optional[str]
    votes: Dict[str, str] = field(default_factory=dict)  # voter_id -> candidate_id
    candidates: Set[str] = field(default_factory=set)
    election_timeout: datetime = field(
        default_factory=lambda: datetime.utcnow() + timedelta(seconds=10)
    )


@dataclass
class DistributedTransaction:
    """Distributed transaction"""

    transaction_id: str
    coordinator_id: str
    participants: Set[str]
    phase: TransactionPhase
    operations: List[Dict[str, Any]]
    votes: Dict[str, bool] = field(default_factory=dict)  # participant_id -> vote
    timeout: datetime = field(default_factory=lambda: datetime.utcnow() + timedelta(seconds=30))
    status: str = "active"


class DistributedCoordinator:
    """
    Distributed Coordination System
    Implements Raft-like consensus, distributed locking, and 2PC transactions
    """

    # Expose enums as class attributes for easier access
    CoordinationState = CoordinationState
    LockState = LockState
    TransactionPhase = TransactionPhase

    def __init__(self, agent_id: str, a2a_communication):
        self.agent_id = agent_id
        self.a2a_comm = a2a_communication
        self.logger = logging.getLogger(f"DistributedCoordinator-{agent_id}")

        # Cache MessageType to avoid circular imports
        self._message_type = None

        # Coordination state
        self.state = CoordinationState.ACTIVE
        self.current_term = 0
        self.voted_for: Optional[str] = None
        self.leader_id: Optional[str] = None
        self.last_heartbeat = datetime.utcnow()

        # Distributed locks
        self.locks: Dict[str, DistributedLock] = {}
        self.lock_requests: Dict[str, Dict[str, Any]] = {}  # request_id -> request_data

        # Leader election
        self.election_data: Optional[LeaderElection] = None
        self.known_agents: Set[str] = set()

        # Distributed transactions
        self.transactions: Dict[str, DistributedTransaction] = {}

        # Configuration
        self.election_timeout = 5.0  # seconds
        self.heartbeat_interval = 2.0  # seconds
        self.lock_timeout = 30.0  # seconds

        # Performance tracking
        self.coordination_metrics = {
            "elections_started": 0,
            "elections_won": 0,
            "locks_acquired": 0,
            "locks_released": 0,
            "transactions_coordinated": 0,
            "consensus_proposals": 0,
        }

        # Synchronization
        self._coordination_lock = asyncio.Lock()
        self._election_lock = asyncio.Lock()
        self._transaction_lock = asyncio.Lock()

        # Background tasks
        self._background_tasks: Set[asyncio.Task] = set()
        self._shutdown = False

        self._background_tasks_started = False

    def _get_message_type(self):
        """Get MessageType class, importing only once to avoid circular imports"""
        if self._message_type is None:
            from .strands_communication import MessageType

            self._message_type = MessageType
        return self._message_type

    async def _start_background_tasks(self):
        """Start background coordination tasks"""
        if self._background_tasks_started:
            return

        try:
            tasks = [
                self._leader_election_monitor(),
                self._lock_timeout_monitor(),
                self._transaction_timeout_monitor(),
                self._coordination_heartbeat(),
            ]

            for task_coro in tasks:
                task = asyncio.create_task(task_coro)
                self._background_tasks.add(task)
                task.add_done_callback(self._background_tasks.discard)

            self._background_tasks_started = True
        except RuntimeError:
            # No event loop - tasks will be started when needed
            pass

    # Leader Election Implementation (Raft-like)
    async def start_leader_election(self) -> bool:
        """Start leader election process"""
        # Ensure background tasks are running
        await self._start_background_tasks()

        async with self._election_lock:
            if self.state == CoordinationState.ELECTING:
                return False  # Already electing

            self.logger.info(f"Starting leader election for term {self.current_term + 1}")

            self.current_term += 1
            self.voted_for = self.agent_id
            self.state = CoordinationState.ELECTING
            self.coordination_metrics["elections_started"] += 1

            # Create election data
            self.election_data = LeaderElection(
                term=self.current_term,
                leader_id=None,
                candidates={self.agent_id},
                votes={self.agent_id: self.agent_id},  # Vote for self
            )

            # Send vote requests to all known agents
            vote_request = {
                "action": "coordination_request",
                "type": "vote_request",
                "data": {
                    "candidate_id": self.agent_id,
                    "term": self.current_term,
                    "election_id": str(uuid.uuid4()),
                },
            }

            # Send to all known agents via A2A
            for agent_id in self.known_agents:
                if agent_id != self.agent_id:
                    try:
                        await self.a2a_comm.send_message(
                            recipient_id=agent_id,
                            message_type=self._get_message_type().REQUEST,
                            payload=vote_request,
                        )
                    except Exception as e:
                        self.logger.error(f"Failed to send vote request to {agent_id}: {e}")

            # Wait for votes or timeout
            await asyncio.sleep(self.election_timeout)
            return await self._count_election_votes()

    async def handle_vote_request(self, sender_id: str, term: int, candidate_id: str) -> bool:
        """Handle incoming vote request"""
        async with self._election_lock:
            # Check term validity
            if term < self.current_term:
                self.logger.debug(f"Rejecting vote for {candidate_id} - stale term {term}")
                return False

            # Update term if newer
            if term > self.current_term:
                self.current_term = term
                self.voted_for = None
                self.state = CoordinationState.FOLLOWING

            # Vote if haven't voted yet or voted for same candidate
            if self.voted_for is None or self.voted_for == candidate_id:
                self.voted_for = candidate_id
                self.logger.info(f"Voting for {candidate_id} in term {term}")
                return True

            return False

    async def handle_vote_response(self, voter_id: str, candidate_id: str, vote: bool, term: int):
        """Handle vote response"""
        if not self.election_data or term != self.current_term:
            return

        if vote:
            self.election_data.votes[voter_id] = candidate_id
            self.logger.debug(f"Received vote from {voter_id} for {candidate_id}")

    async def _count_election_votes(self) -> bool:
        """Count election votes and determine winner"""
        if not self.election_data:
            return False

        # Count votes for each candidate
        vote_counts = defaultdict(int)
        for voter_id, candidate_id in self.election_data.votes.items():
            vote_counts[candidate_id] += 1

        total_agents = len(self.known_agents) + 1  # Include self
        majority_threshold = (total_agents // 2) + 1

        # Check if we won
        our_votes = vote_counts.get(self.agent_id, 0)
        if our_votes >= majority_threshold:
            await self._become_leader()
            return True

        # Check if someone else won
        for candidate_id, votes in vote_counts.items():
            if votes >= majority_threshold:
                self.leader_id = candidate_id
                self.state = CoordinationState.FOLLOWING
                self.logger.info(f"Agent {candidate_id} elected as leader with {votes} votes")
                return False

        # No winner - election failed
        self.state = CoordinationState.ACTIVE
        self.election_data = None
        self.logger.warning("Election failed - no majority winner")
        return False

    async def _become_leader(self):
        """Become the leader"""
        self.leader_id = self.agent_id
        self.state = CoordinationState.ACTIVE
        self.coordination_metrics["elections_won"] += 1

        self.logger.info(f"Became leader for term {self.current_term}")

        # Send leader announcement
        announcement = {
            "action": "coordination_request",
            "type": "leader_announcement",
            "data": {"leader_id": self.agent_id, "term": self.current_term},
        }

        for agent_id in self.known_agents:
            if agent_id != self.agent_id:
                try:
                    await self.a2a_comm.send_message(
                        recipient_id=agent_id,
                        message_type=self._get_message_type().BROADCAST,
                        payload=announcement,
                    )
                except Exception as e:
                    self.logger.error(f"Failed to announce leadership to {agent_id}: {e}")

    # Distributed Locking Implementation
    async def acquire_distributed_lock(
        self, resource_id: str, timeout_seconds: float = 30.0
    ) -> bool:
        """Acquire a distributed lock"""
        if not self.is_leader():
            # Forward to leader
            if self.leader_id:
                return await self._request_lock_from_leader(resource_id, timeout_seconds)
            else:
                # No leader - try to become one
                if await self.start_leader_election():
                    return await self.acquire_distributed_lock(resource_id, timeout_seconds)
                return False

        async with self._coordination_lock:
            request_id = str(uuid.uuid4())

            # Check if lock exists and is available
            if resource_id in self.locks:
                lock = self.locks[resource_id]
                if (
                    lock.state == LockState.HELD
                    and lock.expires_at
                    and datetime.utcnow() > lock.expires_at
                ):
                    # Lock expired - release it
                    await self._release_lock_internal(resource_id)

                if lock.state == LockState.HELD:
                    # Lock is held - add to waiters
                    if self.agent_id not in lock.waiters:
                        lock.waiters.append(self.agent_id)
                    return await self._wait_for_lock(resource_id, timeout_seconds)

            # Create new lock or acquire available lock
            lock = DistributedLock(
                resource_id=resource_id,
                holder_id=self.agent_id,
                state=LockState.HELD,
                acquired_at=datetime.utcnow(),
                expires_at=datetime.utcnow() + timedelta(seconds=timeout_seconds),
            )

            self.locks[resource_id] = lock
            self.coordination_metrics["locks_acquired"] += 1

            # Notify other agents about lock acquisition
            await self._broadcast_lock_status(resource_id, "acquired")

            self.logger.info(f"Acquired distributed lock for resource {resource_id}")
            return True

    async def release_distributed_lock(self, resource_id: str) -> bool:
        """Release a distributed lock"""
        if not self.is_leader():
            # Forward to leader
            if self.leader_id:
                return await self._request_lock_release_from_leader(resource_id)
            return False

        return await self._release_lock_internal(resource_id)

    async def _release_lock_internal(self, resource_id: str) -> bool:
        """Internal lock release implementation"""
        async with self._coordination_lock:
            if resource_id not in self.locks:
                return False

            lock = self.locks[resource_id]

            # Only holder can release
            if lock.holder_id != self.agent_id:
                return False

            lock.state = LockState.RELEASED
            lock.holder_id = None
            self.coordination_metrics["locks_released"] += 1

            # Notify waiters
            if lock.waiters:
                next_holder = lock.waiters.pop(0)
                lock.holder_id = next_holder
                lock.state = LockState.HELD
                lock.acquired_at = datetime.utcnow()

                # Notify new holder
                await self._notify_lock_granted(next_holder, resource_id)
            else:
                # No waiters - remove lock
                del self.locks[resource_id]

            # Broadcast lock status change
            await self._broadcast_lock_status(resource_id, "released")

            self.logger.info(f"Released distributed lock for resource {resource_id}")
            return True

    async def _wait_for_lock(self, resource_id: str, timeout_seconds: float) -> bool:
        """Wait for lock to become available"""
        start_time = time.time()

        while time.time() - start_time < timeout_seconds:
            if resource_id in self.locks:
                lock = self.locks[resource_id]
                if lock.holder_id == self.agent_id:
                    return True

            await asyncio.sleep(0.1)

        # Timeout - remove from waiters
        if resource_id in self.locks:
            lock = self.locks[resource_id]
            if self.agent_id in lock.waiters:
                lock.waiters.remove(self.agent_id)

        return False

    # Distributed Transaction Implementation (2PC)
    async def start_distributed_transaction(
        self, participants: List[str], operations: List[Dict[str, Any]]
    ) -> str:
        """Start a distributed transaction using Two-Phase Commit"""
        if not self.is_leader():
            raise Exception("Only leader can coordinate transactions")

        transaction_id = str(uuid.uuid4())

        async with self._transaction_lock:
            transaction = DistributedTransaction(
                transaction_id=transaction_id,
                coordinator_id=self.agent_id,
                participants=set(participants),
                phase=TransactionPhase.PREPARE,
                operations=operations,
            )

            self.transactions[transaction_id] = transaction
            self.coordination_metrics["transactions_coordinated"] += 1

        # Phase 1: Prepare
        success = await self._execute_prepare_phase(transaction_id)

        if success:
            # Phase 2: Commit
            return await self._execute_commit_phase(transaction_id)
        else:
            # Abort transaction
            await self._execute_abort_phase(transaction_id)
            return None

    async def _execute_prepare_phase(self, transaction_id: str) -> bool:
        """Execute prepare phase of 2PC"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return False

        self.logger.info(f"Starting prepare phase for transaction {transaction_id}")

        # Send prepare requests to all participants
        prepare_request = {
            "action": "coordination_request",
            "type": "transaction_prepare",
            "data": {"transaction_id": transaction_id, "operations": transaction.operations},
        }

        for participant_id in transaction.participants:
            try:
                await self.a2a_comm.send_message(
                    recipient_id=participant_id,
                    message_type=self._get_message_type().REQUEST,
                    payload=prepare_request,
                    requires_ack=True,
                )
            except Exception as e:
                self.logger.error(f"Failed to send prepare to {participant_id}: {e}")
                transaction.votes[participant_id] = False

        # Wait for all votes or timeout
        start_time = time.time()
        while time.time() - start_time < 15.0:  # 15 second timeout
            if len(transaction.votes) >= len(transaction.participants):
                break
            await asyncio.sleep(0.1)

        # Check if all participants voted yes
        all_yes = len(transaction.votes) == len(transaction.participants) and all(
            transaction.votes.values()
        )

        if all_yes:
            transaction.phase = TransactionPhase.COMMIT
            return True
        else:
            transaction.phase = TransactionPhase.ABORT
            return False

    async def _execute_commit_phase(self, transaction_id: str) -> str:
        """Execute commit phase of 2PC"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return None

        self.logger.info(f"Starting commit phase for transaction {transaction_id}")

        # Send commit requests to all participants
        commit_request = {
            "action": "coordination_request",
            "type": "transaction_commit",
            "data": {"transaction_id": transaction_id},
        }

        for participant_id in transaction.participants:
            try:
                await self.a2a_comm.send_message(
                    recipient_id=participant_id,
                    message_type=self._get_message_type().REQUEST,
                    payload=commit_request,
                )
            except Exception as e:
                self.logger.error(f"Failed to send commit to {participant_id}: {e}")

        transaction.status = "committed"
        return transaction_id

    async def _execute_abort_phase(self, transaction_id: str):
        """Execute abort phase of 2PC"""
        transaction = self.transactions.get(transaction_id)
        if not transaction:
            return

        self.logger.info(f"Aborting transaction {transaction_id}")

        # Send abort requests to all participants
        abort_request = {
            "action": "coordination_request",
            "type": "transaction_abort",
            "data": {"transaction_id": transaction_id},
        }

        for participant_id in transaction.participants:
            try:
                await self.a2a_comm.send_message(
                    recipient_id=participant_id,
                    message_type=self._get_message_type().REQUEST,
                    payload=abort_request,
                )
            except Exception as e:
                self.logger.error(f"Failed to send abort to {participant_id}: {e}")

        transaction.status = "aborted"

    # Message Handlers
    async def handle_coordination_message(self, message) -> Dict[str, Any]:
        """Handle coordination-related messages"""
        coordination_type = message.payload.get("type")
        data = message.payload.get("data", {})
        sender_id = message.sender_id

        if coordination_type == "vote_request":
            vote = await self.handle_vote_request(sender_id, data["term"], data["candidate_id"])
            return {"vote": vote, "term": self.current_term}

        elif coordination_type == "vote_response":
            await self.handle_vote_response(
                sender_id, data["candidate_id"], data["vote"], data["term"]
            )
            return {"acknowledged": True}

        elif coordination_type == "leader_announcement":
            self.leader_id = data["leader_id"]
            self.current_term = data["term"]
            self.state = CoordinationState.FOLLOWING
            return {"acknowledged": True}

        elif coordination_type == "transaction_prepare":
            can_commit = await self._handle_transaction_prepare(data)
            return {"vote": can_commit, "transaction_id": data["transaction_id"]}

        elif coordination_type == "transaction_commit":
            await self._handle_transaction_commit(data["transaction_id"])
            return {"status": "committed"}

        elif coordination_type == "transaction_abort":
            await self._handle_transaction_abort(data["transaction_id"])
            return {"status": "aborted"}

        return {"error": f"Unknown coordination type: {coordination_type}"}

    # Background Tasks
    async def _leader_election_monitor(self):
        """Monitor leader health and trigger elections"""
        while not self._shutdown:
            try:
                if self.state == CoordinationState.FOLLOWING:
                    # Check if leader is still alive
                    if (
                        datetime.utcnow() - self.last_heartbeat
                    ).total_seconds() > self.election_timeout * 2:
                        self.logger.warning("Leader timeout - starting election")
                        await self.start_leader_election()

                await asyncio.sleep(self.election_timeout / 2)

            except Exception as e:
                self.logger.error(f"Leader election monitor error: {e}")
                await asyncio.sleep(1)

    async def _lock_timeout_monitor(self):
        """Monitor and clean up expired locks"""
        while not self._shutdown:
            try:
                now = datetime.utcnow()
                expired_locks = []

                for resource_id, lock in self.locks.items():
                    if lock.expires_at and now > lock.expires_at:
                        expired_locks.append(resource_id)

                for resource_id in expired_locks:
                    await self._release_lock_internal(resource_id)
                    self.logger.info(f"Released expired lock: {resource_id}")

                await asyncio.sleep(5)  # Check every 5 seconds

            except Exception as e:
                self.logger.error(f"Lock timeout monitor error: {e}")
                await asyncio.sleep(1)

    async def _transaction_timeout_monitor(self):
        """Monitor and clean up expired transactions"""
        while not self._shutdown:
            try:
                now = datetime.utcnow()
                expired_transactions = []

                for tx_id, transaction in self.transactions.items():
                    if now > transaction.timeout:
                        expired_transactions.append(tx_id)

                for tx_id in expired_transactions:
                    await self._execute_abort_phase(tx_id)
                    del self.transactions[tx_id]
                    self.logger.warning(f"Aborted expired transaction: {tx_id}")

                await asyncio.sleep(10)  # Check every 10 seconds

            except Exception as e:
                self.logger.error(f"Transaction timeout monitor error: {e}")
                await asyncio.sleep(1)

    async def _coordination_heartbeat(self):
        """Send coordination heartbeats"""
        while not self._shutdown:
            try:
                if self.is_leader():
                    # Send heartbeats to followers
                    heartbeat = {
                        "action": "coordination_request",
                        "type": "leader_heartbeat",
                        "data": {"leader_id": self.agent_id, "term": self.current_term},
                    }

                    for agent_id in self.known_agents:
                        if agent_id != self.agent_id:
                            try:
                                await self.a2a_comm.send_message(
                                    recipient_id=agent_id,
                                    message_type=self._get_message_type().HEARTBEAT,
                                    payload=heartbeat,
                                )
                            except Exception as e:
                                self.logger.debug(f"Heartbeat failed to {agent_id}: {e}")

                await asyncio.sleep(self.heartbeat_interval)

            except Exception as e:
                self.logger.error(f"Coordination heartbeat error: {e}")
                await asyncio.sleep(1)

    # Utility Methods
    def is_leader(self) -> bool:
        """Check if this agent is the current leader"""
        return self.leader_id == self.agent_id and self.state == CoordinationState.ACTIVE

    def add_known_agent(self, agent_id: str):
        """Add an agent to the known agents set"""
        self.known_agents.add(agent_id)

    def remove_known_agent(self, agent_id: str):
        """Remove an agent from the known agents set"""
        self.known_agents.discard(agent_id)

    def get_coordination_status(self) -> Dict[str, Any]:
        """Get current coordination status"""
        return {
            "agent_id": self.agent_id,
            "state": self.state.value,
            "current_term": self.current_term,
            "leader_id": self.leader_id,
            "is_leader": self.is_leader(),
            "known_agents": len(self.known_agents),
            "active_locks": len(self.locks),
            "active_transactions": len(self.transactions),
            "metrics": self.coordination_metrics,
        }

    async def shutdown(self):
        """Shutdown the coordination system"""
        self.logger.info("Shutting down distributed coordinator")
        self._shutdown = True

        # Cancel background tasks
        for task in self._background_tasks:
            task.cancel()

        if self._background_tasks:
            await asyncio.gather(*self._background_tasks, return_exceptions=True)

        # Release all held locks
        for resource_id in list(self.locks.keys()):
            if self.locks[resource_id].holder_id == self.agent_id:
                await self._release_lock_internal(resource_id)

        self.logger.info("Distributed coordinator shutdown complete")

    # Helper methods for forwarding requests
    async def _request_lock_from_leader(self, resource_id: str, timeout_seconds: float) -> bool:
        """Request lock from leader"""
        if not self.leader_id:
            return False

        lock_request = {
            "action": "coordination_request",
            "type": "lock_request",
            "data": {
                "resource_id": resource_id,
                "timeout": timeout_seconds,
                "requester_id": self.agent_id,
            },
        }

        try:
            # Import MessageType from strands_communication to avoid circular import
            from .strands_communication import MessageType

            await self.a2a_comm.send_message(
                recipient_id=self.leader_id, message_type=MessageType.REQUEST, payload=lock_request
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to request lock from leader: {e}")
            return False

    async def _request_lock_release_from_leader(self, resource_id: str) -> bool:
        """Request lock release from leader"""
        if not self.leader_id:
            return False

        release_request = {
            "action": "coordination_request",
            "type": "lock_release",
            "data": {"resource_id": resource_id, "requester_id": self.agent_id},
        }

        try:
            await self.a2a_comm.send_message(
                recipient_id=self.leader_id,
                message_type=self._get_message_type().REQUEST,
                payload=release_request,
            )
            return True
        except Exception as e:
            self.logger.error(f"Failed to request lock release from leader: {e}")
            return False

    async def _broadcast_lock_status(self, resource_id: str, status: str):
        """Broadcast lock status change"""
        lock_status = {
            "action": "coordination_request",
            "type": "lock_status",
            "data": {
                "resource_id": resource_id,
                "status": status,
                "holder_id": self.locks.get(resource_id, {}).holder_id
                if resource_id in self.locks
                else None,
            },
        }

        for agent_id in self.known_agents:
            if agent_id != self.agent_id:
                try:
                    await self.a2a_comm.send_message(
                        recipient_id=agent_id,
                        message_type=self._get_message_type().BROADCAST,
                        payload=lock_status,
                    )
                except Exception as e:
                    self.logger.debug(f"Failed to broadcast lock status to {agent_id}: {e}")

    async def _notify_lock_granted(self, agent_id: str, resource_id: str):
        """Notify agent that lock has been granted"""
        lock_granted = {
            "action": "coordination_request",
            "type": "lock_granted",
            "data": {"resource_id": resource_id},
        }

        try:
            await self.a2a_comm.send_message(
                recipient_id=agent_id,
                message_type=self._get_message_type().RESPONSE,
                payload=lock_granted,
            )
        except Exception as e:
            self.logger.error(f"Failed to notify lock granted to {agent_id}: {e}")

    async def _handle_transaction_prepare(self, data: Dict[str, Any]) -> bool:
        """Handle transaction prepare request with real validation"""
        transaction_id = data["transaction_id"]
        operations = data["operations"]

        self.logger.info(
            f"Preparing transaction {transaction_id} with {len(operations)} operations"
        )

        try:
            # Real validation logic
            for operation in operations:
                # Validate operation structure
                if not all(key in operation for key in ["type", "resource", "action"]):
                    self.logger.error(f"Invalid operation structure: {operation}")
                    return False

                # Check resource availability
                resource_id = operation["resource"]
                action = operation["action"]

                # For read operations, always allow
                if action in ["read", "query", "get"]:
                    continue

                # For write operations, check if resource is locked
                if resource_id in self.locks:
                    lock = self.locks[resource_id]
                    if lock.state == LockState.HELD and lock.holder_id != self.agent_id:
                        self.logger.error(f"Resource {resource_id} is locked by {lock.holder_id}")
                        return False

                # Validate operation type
                if operation["type"] not in ["memory", "database", "file", "network"]:
                    self.logger.error(f"Unknown operation type: {operation['type']}")
                    return False

                # Check agent capabilities
                required_capability = f"{operation['type']}_{action}"
                # Note: In real implementation, would check against agent capabilities

            # All validations passed
            self.logger.info(f"Transaction {transaction_id} prepare: APPROVED")
            return True

        except Exception as e:
            self.logger.error(f"Transaction {transaction_id} prepare error: {e}")
            return False

    async def _handle_transaction_commit(self, transaction_id: str):
        """Handle transaction commit request"""
        self.logger.info(f"Committing transaction {transaction_id}")
        # Implementation would execute the committed operations

    async def _handle_transaction_abort(self, transaction_id: str):
        """Handle transaction abort request"""
        self.logger.info(f"Aborting transaction {transaction_id}")
        # Implementation would rollback any prepared changes
