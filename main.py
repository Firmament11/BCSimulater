# -*- coding: utf-8 -*-

import hashlib
import json
import time
import threading
import random
import os
import logging
from typing import List, Dict, Tuple, Optional
import tkinter as tk
from tkinter import ttk, messagebox, scrolledtext
from ecdsa import SigningKey, SECP256k1, VerifyingKey, BadSignatureError
from queue import Queue, Empty

# =========================
# 加密相关辅助函数
# =========================

def generate_key_pair() -> Tuple[SigningKey, VerifyingKey]:
    """
    生成一对ECDSA密钥
    """
    private_key = SigningKey.generate(curve=SECP256k1)
    public_key = private_key.get_verifying_key()
    return private_key, public_key

# =========================
# 交易相关类和函数
# =========================

class TransactionInput:
    """
    交易输入类
    """

    def __init__(self, tx_id: str, output_index: int, signature: str = "", public_key: str = ""):
        self.tx_id = tx_id  # 引用的交易ID
        self.output_index = output_index  # 引用交易输出的索引
        self.signature = signature  # 签名
        self.public_key = public_key  # 公钥

    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'output_index': self.output_index,
            'signature': self.signature,
            'public_key': self.public_key
        }


class TransactionOutput:
    """
    交易输出类
    """

    def __init__(self, address: str, amount: float):
        self.address = address  # 接收方地址
        self.amount = amount  # 金额

    def to_dict(self) -> Dict:
        return {
            'address': self.address,
            'amount': self.amount
        }


class Transaction:
    """
    交易类
    """

    def __init__(self, inputs: List[TransactionInput], outputs: List[TransactionOutput]):
        self.inputs = inputs
        self.outputs = outputs
        self.tx_id = self.calculate_tx_id()
        self.fee = 0.0  # 初始化交易费

    def calculate_tx_hash(self, include_signatures=True) -> str:
        """
        计算交易哈希值。可以选择是否包含签名。
        """
        if include_signatures:
            tx_contents = json.dumps([inp.to_dict() for inp in self.inputs] +
                                     [out.to_dict() for out in self.outputs],
                                     sort_keys=True).encode()
        else:
            # 排除签名用于签名和验证
            tx_contents = json.dumps([{'tx_id': inp.tx_id, 'output_index': inp.output_index} for inp in self.inputs] +
                                     [out.to_dict() for out in self.outputs],
                                     sort_keys=True).encode()
        return hashlib.sha256(tx_contents).hexdigest()

    def calculate_tx_id(self) -> str:
        """
        计算交易ID（包含签名）。
        """
        return self.calculate_tx_hash(include_signatures=True)

    def calculate_fee(self, utxo_set: Dict[Tuple[str, int], Dict[str, float]]) -> float:
        """
        计算交易费
        """
        input_amount = 0.0
        for inp in self.inputs:
            utxo = utxo_set.get((inp.tx_id, inp.output_index))
            if utxo:
                input_amount += utxo['amount']
        output_amount = sum(out.amount for out in self.outputs)
        fee = input_amount - output_amount
        self.fee = fee
        return fee

    def sign_input(self, input_index: int, private_key: SigningKey):
        """
        对指定输入进行签名。
        """
        # 先设置公钥
        self.inputs[input_index].public_key = private_key.get_verifying_key().to_string().hex()
        print(f"交易 {self.tx_id} 的输入 {input_index} 设置公钥: {self.inputs[input_index].public_key}")

        # 计算不包含签名的哈希
        tx_hash = self.calculate_tx_hash(include_signatures=False).encode()
        print(f"Signing tx_hash: {tx_hash}")

        # 生成签名
        signature = private_key.sign(tx_hash).hex()
        self.inputs[input_index].signature = signature
        print(f"交易 {self.tx_id} 的输入 {input_index} 签名: {signature}")

        # 更新交易ID以包含签名
        self.tx_id = self.calculate_tx_id()
        print(f"交易 {self.tx_id} 的输入 {input_index} 已签名并更新 tx_id")

    def verify_input(self, input_index: int) -> bool:
        """
        验证指定输入的签名。
        """
        if self.inputs[input_index].signature == "" or self.inputs[input_index].public_key == "":
            return False
        tx_hash = self.calculate_tx_hash(include_signatures=False).encode()
        print(f"Verifying tx_hash: {tx_hash}")

        signature = bytes.fromhex(self.inputs[input_index].signature)
        public_key_hex = self.inputs[input_index].public_key
        try:
            public_key = VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=SECP256k1)
        except Exception as e:
            print(f"公钥格式错误: {e}")
            return False
        try:
            result = public_key.verify(signature, tx_hash)
            print(f"交易 {self.tx_id} 的输入 {input_index} 签名验证结果: {result}")
            return result
        except BadSignatureError:
            return False

    def to_dict(self) -> Dict:
        return {
            'tx_id': self.tx_id,
            'inputs': [inp.to_dict() for inp in self.inputs],
            'outputs': [out.to_dict() for out in self.outputs]
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)


def create_coinbase_transaction(miner_address: str, block_height: int, block_reward: float, transaction_fees: float = 0.0) -> Transaction:
    """
    创建创币交易（Coinbase Transaction）
    """
    output = TransactionOutput(address=miner_address, amount=block_reward + transaction_fees)
    tx = Transaction(inputs=[], outputs=[output])
    tx.fee = 0.0  # 创币交易不涉及费用
    print(f"创建创币交易: Tx ID={tx.tx_id}, 地址={miner_address}, 金额={block_reward} BTC, 交易费={transaction_fees} BTC")
    return tx


def create_transaction(sender_private_key: SigningKey, sender_address: str,
                       recipient_address: str, amount: float,
                       utxo_set: Dict[Tuple[str, int], Dict[str, float]],
                       fee: float = 0.0) -> Transaction:
    """
    创建并签名一笔交易，包含交易费用
    """
    total_required = amount + fee
    inputs = []
    outputs = [TransactionOutput(address=recipient_address, amount=amount)]
    total_input = 0.0

    # 遍历UTXO集寻找足够的余额
    sorted_utxos = sorted(utxo_set.items(), key=lambda item: item[1]['amount'], reverse=True)
    for (tx_id, output_index), utxo in sorted_utxos:
        if utxo['address'] == sender_address:
            inputs.append(TransactionInput(tx_id=tx_id, output_index=output_index))
            total_input += utxo['amount']
            print(f"选择UTXO: Tx ID={tx_id}, 输出索引={output_index}, 金额={utxo['amount']} BTC")
            if total_input >= total_required:
                break

    if total_input < total_required:
        raise ValueError("余额不足")

    change = total_input - total_required
    if change > 0:
        outputs.append(TransactionOutput(address=sender_address, amount=change))
        print(f"添加找零: 地址={sender_address}, 金额={change} BTC")

    tx = Transaction(inputs=inputs, outputs=outputs)
    for i in range(len(inputs)):
        tx.sign_input(i, sender_private_key)
    # Verify all inputs
    for i in range(len(inputs)):
        if not tx.verify_input(i):
            raise ValueError("签名验证失败")
    # Set fee
    tx.calculate_fee(utxo_set)
    print(f"创建交易成功: Tx ID={tx.tx_id}, 交易费={tx.fee} BTC")
    return tx

# =========================
# 区块相关类和函数
# =========================

class BlockHeader:
    """
    区块头类
    """

    def __init__(self, version: int, previous_hash: str, merkle_root: str,
                 timestamp: int, bits: str, nonce: int):
        self.version = version
        self.previous_hash = previous_hash
        self.merkle_root = merkle_root
        self.timestamp = timestamp
        self.bits = bits
        self.nonce = nonce

    def to_dict(self) -> Dict:
        return {
            'version': self.version,
            'previous_hash': self.previous_hash,
            'merkle_root': self.merkle_root,
            'timestamp': self.timestamp,
            'bits': self.bits,
            'nonce': self.nonce
        }


class Block:
    """
    区块类
    """

    def __init__(self, version: int, previous_hash: str,
                 transactions: List[Transaction], bits: str):
        self.header = BlockHeader(
            version=version,
            previous_hash=previous_hash,
            merkle_root=self.calculate_merkle_root(transactions),
            timestamp=int(time.time()),
            bits=bits,
            nonce=0
        )
        self.transactions = transactions
        self.hash = self.calculate_hash()

    def calculate_merkle_root(self, transactions: List[Transaction]) -> str:
        """
        计算Merkle Root
        """
        tx_hashes = [tx.tx_id for tx in transactions]
        if not tx_hashes:
            return ''
        while len(tx_hashes) > 1:
            if len(tx_hashes) % 2 != 0:
                tx_hashes.append(tx_hashes[-1])  # 奇数时复制最后一个
            tx_hashes = [hashlib.sha256((tx_hashes[i] + tx_hashes[i + 1]).encode()).hexdigest()
                         for i in range(0, len(tx_hashes), 2)]
        return tx_hashes[0]

    def calculate_hash(self) -> str:
        """
        计算区块哈希（双 SHA-256哈希）
        """
        header_contents = json.dumps(self.header.to_dict(), sort_keys=True).encode()
        first_hash = hashlib.sha256(header_contents).digest()
        second_hash = hashlib.sha256(first_hash).hexdigest()
        return second_hash

    def to_dict(self) -> Dict:
        return {
            'header': self.header.to_dict(),
            'transactions': [tx.to_dict() for tx in self.transactions],
            'hash': self.hash
        }

    def __repr__(self) -> str:
        return json.dumps(self.to_dict(), indent=4)

# =========================
# 网络通信模拟
# =========================

class Network:
    """
    模拟网络，处理节点之间的通信
    """

    def __init__(self):
        self.nodes = []
        self.lock = threading.Lock()

    def register_node(self, node):
        with self.lock:
            self.nodes.append(node)
            print(f"网络注册节点 {node.node_id}")

    def broadcast_transaction(self, tx: Transaction, sender_node):
        with self.lock:
            for node in self.nodes:
                if node != sender_node:
                    node.receive_transaction(tx)

    def broadcast_block(self, block: Block, sender_node):
        with self.lock:
            for node in self.nodes:
                if node != sender_node:
                    node.receive_block(block)

# =========================
# 区块链类
# =========================

class Blockchain:
    """
    区块链类
    """

    def __init__(self, difficulty: int = 2, block_reward: float = 50.0):
        self.chain: List[Block] = []
        self.difficulty = difficulty  # 挖矿难度
        self.block_reward = block_reward  # 区块奖励
        self.utxo_set: Dict[Tuple[str, int], Dict[str, float]] = {}
        self.transactions: Dict[str, Transaction] = {}  # 保存所有交易
        self.lock = threading.RLock()  # 使用递归锁代替普通锁
        self.pending_transactions: List[Transaction] = []

        # 初始化日志记录器
        self.logger = logging.getLogger("Blockchain")

        # 生成创世账户
        self.genesis_private_key, self.genesis_public_key = generate_key_pair()
        self.genesis_address = self.genesis_public_key.to_string().hex()

        self.create_genesis_block()

    def get_genesis_address(self) -> str:
        return self.genesis_address

    def get_genesis_private_key(self) -> SigningKey:
        return self.genesis_private_key

    def create_genesis_block(self):
        """
        创建创世块
        """
        genesis_coinbase = create_coinbase_transaction(self.genesis_address, 0, self.block_reward)
        genesis_block = Block(
            version=1,
            previous_hash="0" * 64,
            transactions=[genesis_coinbase],
            bits='0' * self.difficulty + 'F' * (64 - self.difficulty)
        )
        with self.lock:
            self.chain.append(genesis_block)
            self.add_transaction_to_utxo(genesis_coinbase)
            self.transactions[genesis_coinbase.tx_id] = genesis_coinbase
        print("创世块已创建")
        self.logger.info("创世块已创建")

    def add_transaction_to_utxo(self, tx: Transaction):
        """
        将交易添加到UTXO集合，并保存交易
        """
        if not tx.inputs:  # 创币交易
            for index, output in enumerate(tx.outputs):
                self.utxo_set[(tx.tx_id, index)] = {'address': output.address, 'amount': output.amount}
                print(f"添加UTXO: Tx ID={tx.tx_id}, 输出索引={index}, 地址={output.address}, 金额={output.amount} BTC")
        else:
            for inp in tx.inputs:
                self.utxo_set.pop((inp.tx_id, inp.output_index), None)
                print(f"移除UTXO: Tx ID={inp.tx_id}, 输出索引={inp.output_index}")
            for index, output in enumerate(tx.outputs):
                self.utxo_set[(tx.tx_id, index)] = {'address': output.address, 'amount': output.amount}
                print(f"添加UTXO: Tx ID={tx.tx_id}, 输出索引={index}, 地址={output.address}, 金额={output.amount} BTC")
        # 保存交易
        self.transactions[tx.tx_id] = tx

    def add_transaction(self, tx: Transaction):
        """
        添加交易到待处理交易池
        """
        with self.lock:
            self.pending_transactions.append(tx)
            self.logger.info(f"交易已添加: Tx ID={tx.tx_id}")
            print(f"交易已添加: Tx ID={tx.tx_id}")

    def mine_block(self, miner_address: str, network: Optional[Network] = None, callback=None, sender_node=None):
        """
        挖矿，创建新区块并进行工作量证明
        """
        def mine():
            with self.lock:
                if not self.pending_transactions:
                    self.logger.info("没有待处理的交易，无法挖矿。")
                    print("没有待处理的交易，无法挖矿。")
                    if callback:
                        callback(success=False, message="没有待处理的交易，无法挖矿。")
                    return
                # 选择前10笔交易打包进新区块
                selected_txs = self.pending_transactions[:10]
                # 计算总交易费
                total_fees = sum(tx.fee for tx in selected_txs)
                # 创建挖矿奖励交易（创币交易）
                coinbase_tx = create_coinbase_transaction(miner_address, len(self.chain), self.block_reward, transaction_fees=total_fees)
                transactions = [coinbase_tx] + selected_txs
                # 创建新区块
                new_block = Block(
                    version=1,
                    previous_hash=self.chain[-1].hash,
                    transactions=transactions,
                    bits='0' * self.difficulty + 'F' * (64 - self.difficulty)
                )
                self.logger.info(f"开始挖矿新区块，包含 {len(transactions)} 笔交易...")
                print(f"开始挖矿新区块，包含 {len(transactions)} 笔交易...")

            # 工作量证明（双SHA256）
            while not new_block.hash.startswith('0' * self.difficulty):
                new_block.header.nonce += 1
                new_block.hash = new_block.calculate_hash()

            with self.lock:
                self.chain.append(new_block)
                self.add_transaction_to_utxo(coinbase_tx)
                self.transactions[coinbase_tx.tx_id] = coinbase_tx
                for tx in selected_txs:
                    self.add_transaction_to_utxo(tx)
                self.pending_transactions = self.pending_transactions[10:]
                self.logger.info(f"新区块已挖出: {new_block.hash}")
                print(f"新区块已挖出: {new_block.hash}")
                if network and sender_node:
                    network.broadcast_block(new_block, sender_node=sender_node)
                if callback:
                    callback(success=True, message=new_block.hash)

        # 启动挖矿线程
        mining_thread = threading.Thread(target=mine, daemon=True)
        mining_thread.start()
        print("挖矿线程已启动")

    def receive_block(self, block: Block) -> bool:
        """
        接收并验证新区块。如果有效且更长，则添加到链上。
        """
        with self.lock:
            # 验证前一个哈希
            if block.header.previous_hash != self.chain[-1].hash:
                self.logger.warning("接收到的区块的前一个哈希不匹配。")
                print("接收到的区块的前一个哈希不匹配。")
                return False
            # 验证工作量证明
            if not block.hash.startswith('0' * self.difficulty):
                self.logger.warning("接收到的区块不满足难度要求。")
                print("接收到的区块不满足难度要求。")
                return False
            # 验证Merkle Root
            if block.header.merkle_root != block.calculate_merkle_root(block.transactions):
                self.logger.warning("接收到的区块的Merkle Root不匹配。")
                print("接收到的区块的Merkle Root不匹配。")
                return False
            # 验证所有交易的签名
            for tx in block.transactions:
                if tx.inputs:
                    for idx, inp in enumerate(tx.inputs):
                        sender_address = self.get_address_from_input(inp.tx_id, inp.output_index)
                        public_key_hex = inp.public_key
                        if not public_key_hex:
                            self.logger.warning(f"交易 {tx.tx_id} 缺少公钥。")
                            print(f"交易 {tx.tx_id} 缺少公钥。")
                            return False
                        try:
                            public_key = VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=SECP256k1)
                        except Exception as e:
                            self.logger.warning(f"交易 {tx.tx_id} 的公钥格式错误。")
                            print(f"交易 {tx.tx_id} 的公钥格式错误。")
                            return False
                        if not tx.verify_input(idx):
                            self.logger.warning(f"交易 {tx.tx_id} 的签名验证失败。")
                            print(f"交易 {tx.tx_id} 的签名验证失败。")
                            return False
            # 如果所有验证通过，添加区块
            self.chain.append(block)
            # 更新UTXO集
            for tx in block.transactions:
                self.add_transaction_to_utxo(tx)
                self.transactions[tx.tx_id] = tx
            # 移除已确认的交易
            self.pending_transactions = [tx for tx in self.pending_transactions if tx.tx_id not in [t.tx_id for t in block.transactions]]
            self.logger.info(f"区块 {block.hash} 已添加到链上。")
            print(f"区块 {block.hash} 已添加到链上。")
            return True

    def get_balance(self, address: str) -> float:
        """
        获取指定地址的余额
        """
        with self.lock:
            balance = sum(utxo['amount'] for utxo in self.utxo_set.values() if utxo['address'] == address)
            print(f"地址 {address} 的余额: {balance} BTC")
            return balance

    def validate_chain(self) -> bool:
        """
        验证整个区块链的有效性
        """
        with self.lock:
            for i in range(1, len(self.chain)):
                current = self.chain[i]
                previous = self.chain[i - 1]
                if current.header.previous_hash != previous.hash:
                    self.logger.error(f"区块链无效: 区块 {i} 的前一个哈希不匹配。")
                    print(f"区块链无效: 区块 {i} 的前一个哈希不匹配。")
                    return False
                if not current.hash.startswith('0' * self.difficulty):
                    self.logger.error(f"区块链无效: 区块 {i} 不满足难度要求。")
                    print(f"区块链无效: 区块 {i} 不满足难度要求。")
                    return False
                if current.header.merkle_root != current.calculate_merkle_root(current.transactions):
                    self.logger.error(f"区块链无效: 区块 {i} 的Merkle Root不匹配。")
                    print(f"区块链无效: 区块 {i} 的Merkle Root不匹配。")
                    return False
            self.logger.info("区块链验证通过。")
            print("区块链验证通过。")
            return True

    def save_chain(self, filename: str = "blockchain.json"):
        """
        将区块链保存到JSON文件
        """
        with self.lock:
            with open(filename, 'w') as f:
                json.dump([block.to_dict() for block in self.chain], f, indent=4)
            self.logger.info(f"区块链已保存到 {filename}")
            print(f"区块链已保存到 {filename}")

    def load_chain(self, filename: str = "blockchain.json"):
        """
        从JSON文件加载区块链。
        """
        if not os.path.exists(filename):
            self.logger.warning(f"未找到区块链文件: {filename}")
            print(f"未找到区块链文件: {filename}")
            return
        with open(filename, 'r') as f:
            chain_data = json.load(f)
        with self.lock:
            self.chain = []
            self.utxo_set = {}
            self.pending_transactions = []
            self.transactions = {}
            for block_data in chain_data:
                transactions = []
                for tx_data in block_data['transactions']:
                    inputs = [TransactionInput(**inp) for inp in tx_data['inputs']]
                    outputs = [TransactionOutput(**out) for out in tx_data['outputs']]
                    tx = Transaction(inputs, outputs)
                    tx.calculate_fee(self.utxo_set)
                    transactions.append(tx)
                block = Block(
                    version=block_data['header']['version'],
                    previous_hash=block_data['header']['previous_hash'],
                    transactions=transactions,
                    bits=block_data['header']['bits']
                )
                block.header.nonce = block_data['header']['nonce']
                block.hash = block_data['hash']
                self.chain.append(block)
                for tx in transactions:
                    self.add_transaction_to_utxo(tx)
            self.logger.info(f"区块链已从 {filename} 加载。")
            print(f"区块链已从 {filename} 加载。")

    def get_address_from_input(self, tx_id: str, output_index: int) -> str:
        """
        根据交易ID和输出索引获取地址
        """
        with self.lock:
            # 优先从UTXO中查找
            utxo = self.utxo_set.get((tx_id, output_index))
            if utxo:
                return utxo['address']
            # 如果UTXO已被花费，则从所有交易中查找
            tx = self.transactions.get(tx_id)
            if tx and 0 <= output_index < len(tx.outputs):
                return tx.outputs[output_index].address
            return "未知"

    def __repr__(self) -> str:
        return json.dumps([block.to_dict() for block in self.chain], indent=4)

# =========================
# 自定义日志处理器
# =========================

class GUIHandler(logging.Handler):
    """
    自定义日志处理器，用于将日志消息发送到GUI。
    """

    def __init__(self, callback):
        super().__init__()
        self.callback = callback

    def emit(self, record):
        try:
            log_entry = self.format(record)
            self.callback(log_entry)
        except tk.TclError:
            # GUI组件已被销毁，忽略日志
            pass
        except Exception:
            # 捕获所有其他异常，避免程序崩溃
            self.handleError(record)

# =========================
# 网络节点模拟
# =========================

class Node(threading.Thread):
    """
    节点类，模拟比特币网络中的节点
    """

    def __init__(self, node_id: int, blockchain: Blockchain, network: Network, gui_callback=None):
        super().__init__()
        self.node_id = node_id
        self.blockchain = blockchain
        self.network = network
        self.accounts: Dict[str, Tuple[SigningKey, VerifyingKey]] = {}  # 地址 -> (私钥, 公钥)
        self.running = True
        self.logger = logging.getLogger(f"Node-{self.node_id}")
        self.daemon = True  # 设置为守护线程
        self.transaction_queue = Queue()
        self.block_queue = Queue()
        self.gui_callback = gui_callback

        print(f"节点 {self.node_id} 初始化完成")
        # 生成一个默认账户
        private_key, public_key = generate_key_pair()
        address = public_key.to_string().hex()
        self.accounts[address] = (private_key, public_key)
        self.logger.info(f"节点 {self.node_id} 创建账户: {address}")
        print(f"节点 {self.node_id} 创建账户: {address}")
        # 添加创世账户到节点的账户中
        genesis_address = self.blockchain.get_genesis_address()
        genesis_private_key = self.blockchain.get_genesis_private_key()
        self.accounts[genesis_address] = (genesis_private_key, genesis_private_key.get_verifying_key())
        self.logger.info(f"节点 {self.node_id} 添加创世账户: {genesis_address}")
        print(f"节点 {self.node_id} 添加创世账户: {genesis_address}")
        self.auto_run = False
        self.auto_stop_event = threading.Event()

    def run(self):
        print(f"节点 {self.node_id} 开始运行")
        while self.running:
            try:
                # 处理接收到的交易
                tx = self.transaction_queue.get(timeout=1)
                self.handle_transaction(tx)
            except Empty:
                pass

            try:
                # 处理接收到的区块
                block = self.block_queue.get(timeout=1)
                self.handle_block(block)
            except Empty:
                pass

            # Handle auto_run if enabled
            if self.auto_run and not self.auto_stop_event.is_set():
                # 自动交易和挖矿由GUI控制，不在这里处理
                pass

            time.sleep(0.1)  # 防止CPU占用过高

    def handle_transaction(self, tx: Transaction):
        """
        处理接收到的交易
        """
        with self.blockchain.lock:
            # 验证交易是否已经存在
            if tx.tx_id in self.blockchain.transactions or tx in self.blockchain.pending_transactions:
                return
            # 简单的交易验证
            if tx.inputs:
                for idx, inp in enumerate(tx.inputs):
                    sender_address = self.blockchain.get_address_from_input(inp.tx_id, inp.output_index)
                    # 获取发送者的公钥
                    public_key_hex = inp.public_key
                    if not public_key_hex:
                        self.logger.warning(f"交易 {tx.tx_id} 缺少公钥。")
                        print(f"交易 {tx.tx_id} 缺少公钥。")
                        return
                    try:
                        public_key = VerifyingKey.from_string(bytes.fromhex(public_key_hex), curve=SECP256k1)
                    except Exception as e:
                        self.logger.warning(f"交易 {tx.tx_id} 的公钥格式错误。")
                        print(f"交易 {tx.tx_id} 的公钥格式错误。")
                        return
                    if not tx.verify_input(idx):
                        self.logger.warning(f"交易 {tx.tx_id} 的签名验证失败。")
                        print(f"交易 {tx.tx_id} 的签名验证失败。")
                        return
            # 添加到待处理交易
            self.blockchain.add_transaction(tx)
            self.logger.info(f"节点 {self.node_id} 接收到并添加交易 {tx.tx_id}")
            print(f"节点 {self.node_id} 接收到并添加交易 {tx.tx_id}")
            # 广播交易到网络
            self.network.broadcast_transaction(tx, sender_node=self)
            # 如果有GUI回调，通知GUI更新
            if self.gui_callback:
                self.gui_callback()

    def handle_block(self, block: Block):
        """
        处理接收到的区块
        """
        success = self.blockchain.receive_block(block)
        if success:
            self.logger.info(f"节点 {self.node_id} 接收到并添加区块 {block.hash}")
            print(f"节点 {self.node_id} 接收到并添加区块 {block.hash}")
            # 广播区块到网络
            self.network.broadcast_block(block, sender_node=self)
            # 如果有GUI回调，通知GUI更新
            if self.gui_callback:
                self.gui_callback()

    def receive_transaction(self, tx: Transaction):
        """
        接收网络中广播的交易
        """
        self.transaction_queue.put(tx)

    def receive_block(self, block: Block):
        """
        接收网络中广播的区块
        """
        self.block_queue.put(block)

    def mine_block_via_node(self, callback=None):
        """
        通过节点进行挖矿
        """
        if not self.accounts:
            print(f"节点 {self.node_id} 没有账户用于挖矿")
            return
        miner_address = random.choice(list(self.accounts.keys()))
        self.blockchain.mine_block(miner_address, network=self.network, callback=callback, sender_node=self)

    def create_transaction(self, recipient_address: str, amount: float, fee: float = 0.0) -> Optional[Transaction]:
        """
        创建并广播一笔交易。
        """
        with self.blockchain.lock:
            # 获取所有可用的UTXO
            spendable_utxos = [(k, v) for k, v in self.blockchain.utxo_set.items() if v['address'] in self.accounts]
            if not spendable_utxos:
                self.logger.warning(f"节点 {self.node_id} 没有可用的UTXO进行交易。")
                print(f"节点 {self.node_id} 没有可用的UTXO进行交易。")
                return None

            # 过滤发送者拥有余额的账户
            sender_addresses = [address for address in self.accounts.keys() if self.blockchain.get_balance(address) > 0]
            if not sender_addresses:
                self.logger.warning(f"节点 {self.node_id} 没有任何账户具有余额。")
                print(f"节点 {self.node_id} 没有任何账户具有余额。")
                return None

            # 随机选择一个发送者地址
            sender_address = random.choice(sender_addresses)
            # 获取发送者的UTXO
            sender_utxos = [(k, v) for k, v in spendable_utxos if v['address'] == sender_address]
            if not sender_utxos:
                self.logger.warning(f"节点 {self.node_id} 的发送者地址 {sender_address} 没有可用UTXO。")
                print(f"节点 {self.node_id} 的发送者地址 {sender_address} 没有可用UTXO。")
                return None

            # 随机选择一个UTXO
            (tx_id, output_index), utxo = random.choice(sender_utxos)
            sender_private_key, sender_public_key = self.accounts.get(sender_address)
            if not sender_private_key:
                self.logger.warning(f"节点 {self.node_id} 无法找到发送者地址 {sender_address} 的私钥。")
                print(f"节点 {self.node_id} 无法找到发送者地址 {sender_address} 的私钥。")
                return None

            # 创建交易
            try:
                tx = create_transaction(
                    sender_private_key=sender_private_key,
                    sender_address=sender_address,
                    recipient_address=recipient_address,
                    amount=amount,
                    fee=fee,
                    utxo_set=self.blockchain.utxo_set
                )
                self.blockchain.add_transaction(tx)
                self.logger.info(f"节点 {self.node_id} 创建并广播交易 {tx.tx_id} 从 {sender_address} 到 {recipient_address}")
                print(f"节点 {self.node_id} 创建并广播交易 {tx.tx_id} 从 {sender_address} 到 {recipient_address}")
                # 广播交易到网络
                self.network.broadcast_transaction(tx, sender_node=self)
                return tx
            except ValueError as ve:
                self.logger.warning(f"节点 {self.node_id} 创建交易失败 - {ve}")
                print(f"节点 {self.node_id} 创建交易失败 - {ve}")
                return None

    # =========================
    # 区块链GUI界面
    # =========================

class BlockchainGUI:
    """
    图形用户界面类，使用Tkinter实现
    """

    def __init__(self, blockchain: Blockchain, logger: logging.Logger, nodes: List[Node], network: Network):
        try:
            print("开始初始化GUI")
            self.blockchain = blockchain
            self.logger = logger
            self.nodes = nodes
            self.network = network
            self.node_mapping = {f"Node-{node.node_id}": node for node in nodes}

            self.root = tk.Tk()
            self.root.title("比特币区块链模拟器")
            self.root.geometry("1800x1000+50+50")  # 调整窗口大小以适应更多控制按钮
            print("GUI窗口创建完成")

            # 设置样式
            self.style = ttk.Style()
            self.style.theme_use('clam')

            # 创建主框架
            self.main_frame = ttk.Frame(self.root)
            self.main_frame.pack(fill='both', expand=True)
            print("主框架创建完成")

            # 创建左侧信息显示区域（使用Notebook）
            self.left_notebook = ttk.Notebook(self.main_frame)
            self.left_notebook.pack(side='left', fill='y', padx=10, pady=10)
            print("Notebook创建完成")

            # 创建账户信息标签页
            self.account_frame = ttk.Frame(self.left_notebook)
            self.left_notebook.add(self.account_frame, text='账户信息')
            print("账户信息标签页创建完成")

            self.account_info_label = ttk.Label(self.account_frame, text="账户列表", font=("Arial", 14))
            self.account_info_label.pack(pady=5)

            self.account_info_tree = ttk.Treeview(self.account_frame, columns=("Address", "Balance"), show='headings')
            self.account_info_tree.heading("Address", text="地址")
            self.account_info_tree.heading("Balance", text="余额 (BTC)")
            self.account_info_tree.pack(fill='both', expand=True, pady=5)

            # 创建交易信息标签页
            self.transaction_frame = ttk.Frame(self.left_notebook)
            self.left_notebook.add(self.transaction_frame, text='交易信息')
            print("交易信息标签页创建完成")

            self.transaction_info_label = ttk.Label(self.transaction_frame, text="交易记录", font=("Arial", 14))
            self.transaction_info_label.pack(pady=5)

            self.transaction_info_tree = ttk.Treeview(self.transaction_frame,
                                                      columns=("Tx ID", "From", "To", "Amount", "Fee"), show='headings')
            self.transaction_info_tree.heading("Tx ID", text="交易ID")
            self.transaction_info_tree.heading("From", text="发送者")
            self.transaction_info_tree.heading("To", text="接收者")
            self.transaction_info_tree.heading("Amount", text="金额 (BTC)")
            self.transaction_info_tree.heading("Fee", text="交易费 (BTC)")
            self.transaction_info_tree.pack(fill='both', expand=True, pady=5)

            # 创建区块链详情标签页
            self.blockchain_frame = ttk.Frame(self.left_notebook)
            self.left_notebook.add(self.blockchain_frame, text='区块链详情')
            print("区块链详情标签页创建完成")

            self.blockchain_info_label = ttk.Label(self.blockchain_frame, text="区块链信息", font=("Arial", 14))
            self.blockchain_info_label.pack(pady=5)

            self.blockchain_info_tree = ttk.Treeview(self.blockchain_frame,
                                                     columns=("Block Height", "Hash", "Transactions"), show='headings')
            self.blockchain_info_tree.heading("Block Height", text="区块高度")
            self.blockchain_info_tree.heading("Hash", text="区块哈希")
            self.blockchain_info_tree.heading("Transactions", text="交易数")
            self.blockchain_info_tree.pack(fill='both', expand=True, pady=5)

            # 创建右侧控制和显示区域
            self.right_frame = ttk.Frame(self.main_frame)
            self.right_frame.pack(side='right', fill='both', expand=True, padx=10, pady=10)
            print("右侧控制和显示区域创建完成")

            # 创建节点选择框架
            self.node_selection_frame = ttk.Frame(self.right_frame)
            self.node_selection_frame.pack(fill='x', pady=10)
            print("节点选择框架创建完成")

            ttk.Label(self.node_selection_frame, text="选择节点:").pack(side='left', padx=5)

            self.node_var = tk.StringVar()
            node_ids = list(self.node_mapping.keys())
            self.node_combobox = ttk.Combobox(self.node_selection_frame, textvariable=self.node_var, values=node_ids, state='readonly')
            self.node_combobox.pack(side='left', padx=5)
            if node_ids:
                self.node_combobox.current(0)  # 默认选择第一个节点
                self.current_node_label = ttk.Label(self.node_selection_frame, text=f"当前节点: {node_ids[0]}")
                self.current_node_label.pack(side='left', padx=10)
            else:
                self.current_node_label = ttk.Label(self.node_selection_frame, text="当前节点: 无")
                self.current_node_label.pack(side='left', padx=10)

            self.node_combobox.bind("<<ComboboxSelected>>", self.on_node_selected)

            # 创建按钮框架
            self.button_frame = ttk.Frame(self.right_frame)
            self.button_frame.pack(fill='x', pady=10)
            print("按钮框架创建完成")

            self.create_account_button = ttk.Button(self.button_frame, text="创建账户", command=self.create_account)
            self.create_account_button.pack(side='left', padx=5)

            self.mine_block_button = ttk.Button(self.button_frame, text="挖矿", command=self.mine_block)
            self.mine_block_button.pack(side='left', padx=5)

            self.save_button = ttk.Button(self.button_frame, text="保存区块链", command=self.save_blockchain)
            self.save_button.pack(side='left', padx=5)

            self.load_button = ttk.Button(self.button_frame, text="加载区块链", command=self.load_blockchain)
            self.load_button.pack(side='left', padx=5)

            self.create_tx_button = ttk.Button(self.button_frame, text="创建交易",
                                               command=self.open_create_transaction_window)
            self.create_tx_button.pack(side='left', padx=5)

            # 添加独立控制每个节点的自动行为的按钮和状态指示器
            self.auto_control_buttons = {}
            self.auto_status_labels = {}
            for idx, node in enumerate(self.nodes):
                node_name = f"Node-{node.node_id}"
                frame = ttk.Frame(self.button_frame)
                frame.pack(side='top', fill='x', pady=2)

                btn_text = f"开启 {node_name} 自动交易和挖矿"
                button = ttk.Button(frame, text=btn_text, command=lambda n=node, name=node_name: self.toggle_auto_run(n, name))
                button.pack(side='left', padx=5)

                status_label = ttk.Label(frame, text="自动运行: 关闭", foreground="red")
                status_label.pack(side='left', padx=5)

                self.auto_control_buttons[node_name] = button
                self.auto_status_labels[node_name] = status_label

            # 创建日志显示区域
            self.log_frame = ttk.Frame(self.right_frame)
            self.log_frame.pack(fill='both', expand=True, pady=10)
            print("日志显示区域创建完成")

            self.log_label = ttk.Label(self.log_frame, text="系统日志", font=("Arial", 14))
            self.log_label.pack(pady=5)

            self.log_text = scrolledtext.ScrolledText(self.log_frame, width=80, height=20, state='disabled',
                                                      background='black', foreground='white')
            self.log_text.pack(fill='both', expand=True, pady=5)

            # 设置自定义日志处理器，将日志消息发送到GUI。
            self.gui_handler = GUIHandler(self.append_log)
            self.gui_handler.setFormatter(logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s'))
            logger.addHandler(self.gui_handler)

            # 启动定时更新，增加频率到每1秒
            self.periodic_update()
            print("定时更新已启动")

            # Initial update
            self.update_account_info()
            self.update_transaction_info("系统初始化完成。\n\n")
            self.update_blockchain_info()
            print("初始信息更新完成")
        except Exception as e:
            print(f"GUI初始化失败: {e}")
            self.logger.error(f"GUI初始化失败: {e}", exc_info=True)
            messagebox.showerror("初始化错误", f"GUI初始化失败: {e}")
            raise  # 重新抛出异常以便主程序知道发生了错误

    def append_log(self, message: str):
        """
        将日志消息追加到日志显示区域。
        """
        self.log_text.config(state='normal')
        self.log_text.insert(tk.END, message + "\n")
        self.log_text.yview(tk.END)
        self.log_text.config(state='disabled')

    def toggle_auto_run(self, node: Node, node_name: str):
        """
        切换指定节点的自动交易和挖矿功能
        """
        if hasattr(node, 'auto_run'):
            node.auto_run = not node.auto_run
        else:
            node.auto_run = True

        if node.auto_run:
            self.auto_control_buttons[node_name].config(text=f"关闭 {node_name} 自动交易和挖矿")
            self.auto_status_labels[node_name].config(text="自动运行: 开启", foreground="green")
            self.update_transaction_info(f"已开启 {node_name} 的自动交易和挖矿功能。\n\n")
            self.logger.info(f"{node_name} 的自动交易和挖矿功能已开启。")
            print(f"{node_name} 的自动交易和挖矿功能已开启。")
            # 启动自动交易和挖矿线程
            node.auto_thread = threading.Thread(target=self.auto_run_node, args=(node,), daemon=True)
            node.auto_thread.start()
        else:
            self.auto_control_buttons[node_name].config(text=f"开启 {node_name} 自动交易和挖矿")
            self.auto_status_labels[node_name].config(text="自动运行: 关闭", foreground="red")
            self.update_transaction_info(f"已关闭 {node_name} 的自动交易和挖矿功能。\n\n")
            self.logger.info(f"{node_name} 的自动交易和挖矿功能已关闭。")
            print(f"{node_name} 的自动交易和挖矿功能已关闭。")
            # 停止自动线程
            node.auto_stop_event.set()

    def auto_run_node(self, node: Node):
        """
        自动运行节点的交易和挖矿功能
        """
        while not node.auto_stop_event.is_set():
            action = random.choice(['create_tx', 'mine'])
            print(f"节点 {node.node_id} 执行自动操作: {action}")
            if action == 'create_tx':
                # 随机选择一个接收者
                recipient_address = random.choice(self.get_all_accounts())
                if not recipient_address:
                    continue
                # 随机金额
                amount = round(random.uniform(0.1, 1.0), 8)
                node.create_transaction(recipient_address=recipient_address, amount=amount, fee=0.001)
            elif action == 'mine':
                node.mine_block_via_node(callback=self.on_mine_complete_threadsafe)
            time.sleep(random.uniform(5, 10))  # 随机休眠5到10秒

    def on_mine_complete_threadsafe(self, success: bool, message: str, progress_window=None):
        """
        挖矿完成后的回调函数
        """
        def update():
            if success:
                print("挖矿完成，更新GUI")
                self.update_account_info()
                self.update_transaction_info(f"矿工挖出了一个新块。\nHash: {message}\n\n")
                self.logger.info(f"矿工成功挖出一个新块: {message}")
            else:
                print(f"挖矿失败: {message}")
                self.update_transaction_info(f"挖矿失败: {message}\n\n")
                self.logger.warning(f"挖矿失败: {message}")
            if progress_window:
                progress_window.destroy()

        self.root.after(0, update)

    def on_node_selected(self, event):
        selected_node_id = self.node_var.get()
        self.current_node_label.config(text=f"当前节点: {selected_node_id}")
        self.logger.info(f"切换到节点: {selected_node_id}")
        print(f"切换到节点: {selected_node_id}")
        self.update_account_info()
        self.update_transaction_info("节点已切换。\n\n")

    def get_selected_node(self) -> Optional[Node]:
        selected_node_id = self.node_var.get()
        return self.node_mapping.get(selected_node_id)

    def create_account(self):
        """
        创建新账户，并通过交易将部分余额转移给新账户。
        """
        try:
            print("开始创建新账户")
            selected_node = self.get_selected_node()
            if not selected_node:
                messagebox.showerror("错误", "未选择节点。")
                return

            private_key, public_key = generate_key_pair()
            # 使用 to_string().hex() 获取公钥的十六进制表示
            address = public_key.to_string().hex()
            selected_node.accounts[address] = (private_key, public_key)
            self.logger.info(f"节点 {selected_node.node_id} 创建新账户: {address}")
            print(f"节点 {selected_node.node_id} 创建新账户: {address}")

            with selected_node.blockchain.lock:
                # 获取所有可用的UTXO，属于节点的账户
                spendable_utxos = [(k, v) for k, v in selected_node.blockchain.utxo_set.items() if v['address'] in selected_node.accounts]
                if not spendable_utxos:
                    messagebox.showerror("错误", "没有可用的UTXO进行账户创建转账。")
                    self.logger.error("没有可用的UTXO进行账户创建转账。")
                    print("没有可用的UTXO进行账户创建转账。")
                    return

                # 随机选择一个UTXO
                selected_utxo = random.choice(spendable_utxos)
                (tx_id, output_index), utxo = selected_utxo
                sender_address = utxo['address']
                sender_private_key, sender_public_key = selected_node.accounts.get(sender_address)
                if not sender_private_key:
                    messagebox.showerror("错误", "发送方私钥未找到。")
                    self.logger.error(f"发送方地址 {sender_address} 的私钥未找到。")
                    print(f"发送方地址 {sender_address} 的私钥未找到。")
                    return

                # 设置转账金额和交易费
                amount = round(min(utxo['amount'] * 0.5, selected_node.blockchain.block_reward / 2), 8)
                fee = 0.001  # 固定交易费
                print(f"转账金额: {amount} BTC, 交易费: {fee} BTC")

                # 创建交易
                tx = create_transaction(
                    sender_private_key=sender_private_key,
                    sender_address=sender_address,
                    recipient_address=address,
                    amount=amount,
                    fee=fee,
                    utxo_set=selected_node.blockchain.utxo_set
                )
                selected_node.blockchain.add_transaction(tx)
                self.logger.info(
                    f"节点 {selected_node.node_id} 已创建交易: Tx ID={tx.tx_id}, 从={sender_address}, 到={address}, 金额: {amount} BTC, 交易费: {fee} BTC")
                print(f"节点 {selected_node.node_id} 已创建交易: Tx ID={tx.tx_id}, 从={sender_address} 到={address}, 金额: {amount} BTC, 交易费: {fee} BTC")

            # 广播交易到网络
            self.network.broadcast_transaction(tx, sender_node=selected_node)

            self.update_account_info()
            self.update_transaction_info(f"节点 {selected_node.node_id} 创建新账户成功:\n地址: {address}\n余额: {amount} BTC\n\n")
            print("新账户创建成功")
        except ValueError as ve:
            print(f"创建账户失败: {ve}")
            self.logger.error(f"创建账户失败: {ve}")
            messagebox.showerror("错误", f"创建账户失败: {ve}")
        except Exception as e:
            print(f"创建账户失败: {e}")
            self.logger.error(f"创建账户失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"创建账户失败: {e}")

    def mine_block(self):
        """
        手动发起挖矿操作。
        """
        try:
            selected_node = self.get_selected_node()
            if not selected_node:
                messagebox.showerror("错误", "未选择节点。")
                return
            miner_address = random.choice(list(selected_node.accounts.keys())) if selected_node.accounts else None
            if not miner_address:
                messagebox.showerror("错误", "选定节点没有账户可用于挖矿。")
                return
            print(f"选择矿工地址: {miner_address}")

            # 创建挖矿进度窗口
            progress_window = tk.Toplevel(self.root)
            progress_window.title("挖矿中...")
            progress_window.geometry("400x100")
            ttk.Label(progress_window, text=f"节点 {selected_node.node_id} 矿工 {miner_address} 挖矿中...").pack(pady=10)
            progress_bar = ttk.Progressbar(progress_window, mode='indeterminate')
            progress_bar.pack(pady=10, padx=20, fill='x')
            progress_bar.start(10)
            print("挖矿进度窗口已创建")

            # 挖矿线程
            def mine():
                try:
                    print("开始挖矿线程")
                    # 使用回调函数在挖矿完成后更新GUI
                    selected_node.mine_block_via_node(callback=lambda success, message: self.on_mine_complete_threadsafe(success, message, progress_window))
                except Exception as e:
                    print(f"挖矿线程失败: {e}")
                    self.logger.error(f"挖矿线程失败: {e}", exc_info=True)
                    messagebox.showerror("错误", f"挖矿失败: {e}")
                    progress_bar.stop()
                    progress_window.destroy()

            threading.Thread(target=mine, daemon=True).start()
            print("挖矿线程已启动")
        except Exception as e:
            print(f"挖矿失败: {e}")
            self.logger.error(f"挖矿失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"挖矿失败: {e}")

    def save_blockchain(self):
        """
        保存当前区块链到文件。
        """
        try:
            print("保存区块链到文件")
            self.blockchain.save_chain()
            self.update_transaction_info("区块链已保存。\n\n")
            self.logger.info("区块链已保存到文件。")
            print("区块链已保存到文件")
        except Exception as e:
            print(f"保存区块链失败: {e}")
            self.logger.error(f"保存区块链失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"保存区块链失败: {e}")

    def load_blockchain(self):
        """
        从文件加载区块链。
        """
        try:
            print("从文件加载区块链")
            self.blockchain.load_chain()
            self.update_account_info()
            self.update_transaction_info("区块链已加载。\n\n")
            self.update_blockchain_info()
            print("区块链已从文件加载")
        except Exception as e:
            print(f"加载区块链失败: {e}")
            self.logger.error(f"加载区块链失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"加载区块链失败: {e}")

    def open_create_transaction_window(self):
        """
        打开创建交易的窗口。
        """
        selected_node = self.get_selected_node()
        if not selected_node or not selected_node.accounts:
            print("请先为选定的节点创建至少一个账户")
            messagebox.showerror("错误", "请先为选定的节点创建至少一个账户。")
            return

        # 获取当前节点有余额的账户
        accounts_with_balance = [address for address in selected_node.accounts.keys() if self.blockchain.get_balance(address) > 0]
        if not accounts_with_balance:
            print("没有账户具有足够的余额进行交易")
            messagebox.showerror("错误", "没有账户具有足够的余额进行交易。")
            return

        self.tx_window = tk.Toplevel(self.root)
        self.tx_window.title("创建交易")
        self.tx_window.geometry("500x400")
        print("创建交易窗口已打开")

        # 发送者地址
        ttk.Label(self.tx_window, text="发送者地址:").pack(pady=5)
        self.sender_var = tk.StringVar()
        sender_menu = ttk.Combobox(self.tx_window, textvariable=self.sender_var, values=accounts_with_balance,
                                   state='readonly', width=60)
        sender_menu.pack(pady=5)

        # 接收者地址
        ttk.Label(self.tx_window, text="接收者地址:").pack(pady=5)
        self.recipient_var = tk.StringVar()
        recipient_accounts = self.get_all_accounts()
        recipient_menu = ttk.Combobox(self.tx_window, textvariable=self.recipient_var, values=recipient_accounts,
                                      state='readonly', width=60)
        recipient_menu.pack(pady=5)
        if recipient_accounts:
            recipient_menu.current(0)
        else:
            recipient_menu.set("无可用接收者")

        # 金额
        ttk.Label(self.tx_window, text="金额 (BTC):").pack(pady=5)
        self.amount_var = tk.DoubleVar()
        self.amount_entry = ttk.Entry(self.tx_window, textvariable=self.amount_var, width=60)
        self.amount_entry.pack(pady=5)

        # 交易费
        ttk.Label(self.tx_window, text="交易费 (BTC):").pack(pady=5)
        self.fee_var = tk.DoubleVar(value=0.001)  # 默认交易费
        self.fee_entry = ttk.Entry(self.tx_window, textvariable=self.fee_var, width=60)
        self.fee_entry.pack(pady=5)

        # 创建交易按钮
        self.create_tx_button = ttk.Button(self.tx_window, text="创建交易", command=self.create_transaction_window)
        self.create_tx_button.pack(pady=20)
        print("创建交易按钮已添加到窗口")

    def create_transaction_window(self):
        """
        创建并提交一笔交易。
        """
        selected_node = self.get_selected_node()
        if not selected_node:
            messagebox.showerror("错误", "未选择节点。")
            return

        sender_address = self.sender_var.get().strip()
        recipient_address = self.recipient_var.get().strip()
        try:
            amount = float(self.amount_var.get())
            fee = float(self.fee_var.get())
        except ValueError:
            print("金额和交易费必须是数字")
            messagebox.showerror("错误", "金额和交易费必须是数字。")
            return

        if sender_address not in selected_node.accounts:
            print("发送者地址不存在")
            messagebox.showerror("错误", "发送者地址不存在。")
            return
        if recipient_address not in self.get_all_accounts():
            print("接收者地址不存在")
            messagebox.showerror("错误", "接收者地址不存在。")
            return
        if amount <= 0:
            print("金额必须大于0")
            messagebox.showerror("错误", "金额必须大于0。")
            return
        if fee < 0:
            print("交易费不能为负")
            messagebox.showerror("错误", "交易费不能为负。")
            return

        sender_private_key, sender_public_key = selected_node.accounts.get(sender_address)
        if not sender_private_key:
            print("发送者私钥不存在")
            messagebox.showerror("错误", "发送者私钥不存在。")
            return

        try:
            print(f"创建交易: 从={sender_address}, 到={recipient_address}, 金额={amount} BTC, 交易费={fee} BTC")
            tx = create_transaction(
                sender_private_key=sender_private_key,
                sender_address=sender_address,
                recipient_address=recipient_address,
                amount=amount,
                fee=fee,
                utxo_set=selected_node.blockchain.utxo_set
            )
            selected_node.blockchain.add_transaction(tx)
            self.logger.info(
                f"交易 {tx.tx_id} 创建成功: 从 {sender_address} 到 {recipient_address}, 金额: {amount} BTC, 交易费: {fee} BTC.")
            print(f"交易 {tx.tx_id} 创建成功: 从 {sender_address} 到 {recipient_address}, 金额: {amount} BTC, 交易费: {fee} BTC.")
            # 广播交易到网络
            self.network.broadcast_transaction(tx, sender_node=selected_node)
            self.update_transaction_info(
                f"创建交易成功:\nTx ID: {tx.tx_id}\n从: {sender_address}\n到: {recipient_address}\n金额: {amount} BTC\n交易费: {fee} BTC\n\n")
            self.tx_window.destroy()
            # 立即更新账户信息
            self.update_account_info()
        except ValueError as ve:
            print(f"创建交易失败: {ve}")
            self.logger.error(f"创建交易失败: {ve}")
            messagebox.showerror("错误", f"创建交易失败: {ve}")
        except Exception as e:
            print(f"创建交易失败: {e}")
            self.logger.error(f"创建交易失败: {e}", exc_info=True)
            messagebox.showerror("错误", f"创建交易失败: {e}")

    def update_account_info(self):
        """
        更新账户信息显示。
        """
        print("更新账户信息")
        self.account_info_tree.delete(*self.account_info_tree.get_children())
        all_accounts = self.get_all_accounts()
        for address in all_accounts:
            balance = self.blockchain.get_balance(address)
            self.account_info_tree.insert('', 'end', values=(address, round(balance, 8)))
            self.logger.info(f"账户更新: 地址={address}, 余额={balance} BTC")
            print(f"账户更新: 地址={address}, 余额={balance} BTC")

    def update_transaction_info(self, message: str):
        """
        更新交易信息显示，包括已确认的交易和待处理的交易。
        """
        print("更新交易信息")
        self.transaction_info_tree.delete(*self.transaction_info_tree.get_children())
        with self.blockchain.lock:
            # 显示已确认的交易
            for block in self.blockchain.chain:
                for tx in block.transactions:
                    if not tx.inputs:
                        sender = "Coinbase"  # 创币交易
                    else:
                        # 获取发送者地址
                        input_tx = tx.inputs[0]
                        sender = self.blockchain.get_address_from_input(input_tx.tx_id, input_tx.output_index)
                    for out in tx.outputs:
                        # 计算交易费，仅在普通交易中显示
                        fee = tx.fee if tx.inputs else 0.0
                        self.transaction_info_tree.insert('', 'end',
                                                          values=(tx.tx_id, sender, out.address, round(out.amount, 8), round(fee, 8)))
            # 显示待处理的交易
            for tx in self.blockchain.pending_transactions:
                if not tx.inputs:
                    sender = "Coinbase"  # 创币交易
                else:
                    input_tx = tx.inputs[0]
                    sender = self.blockchain.get_address_from_input(input_tx.tx_id, input_tx.output_index)
                for out in tx.outputs:
                    fee = tx.fee if tx.inputs else 0.0
                    self.transaction_info_tree.insert('', 'end',
                                                      values=(tx.tx_id, sender, out.address, round(out.amount, 8), round(fee, 8)))
        # 在日志区域显示详细信息
        if message:
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, message)
            self.log_text.yview(tk.END)
            self.log_text.config(state='disabled')
        print("交易信息更新完成")

    def update_blockchain_info(self):
        """
        更新区块链信息显示。
        """
        print("更新区块链信息")
        self.blockchain_info_tree.delete(*self.blockchain_info_tree.get_children())
        with self.blockchain.lock:
            for i, block in enumerate(self.blockchain.chain):
                # 交易数不包括Coinbase交易
                tx_count = len(block.transactions) - 1 if len(block.transactions) > 0 else 0
                self.blockchain_info_tree.insert('', 'end', values=(i, block.hash, tx_count))
                self.logger.info(f"区块更新: 高度={i}, 哈希={block.hash}, 交易数={tx_count}")
                print(f"区块更新: 高度={i}, 哈希={block.hash}, 交易数={tx_count}")

    def periodic_update(self):
        """
        定期更新GUI显示内容。
        """
        try:
            print("执行定时更新")
            self.update_account_info()
            self.update_transaction_info("")  # 传递空消息，以便只更新交易树
            self.update_blockchain_info()
            # 日志已通过 GUIHandler 实时更新，无需额外处理
        except Exception as e:
            print(f"定时更新失败: {e}")
            self.logger.error(f"定时更新失败: {e}", exc_info=True)
            self.log_text.config(state='normal')
            self.log_text.insert(tk.END, f"定时更新失败: {e}\n")
            self.log_text.config(state='disabled')
        finally:
            # 每1秒调用一次自身，增加更新频率
            self.root.after(1000, self.periodic_update)

    def run(self):
        """
        运行GUI主循环。
        """
        try:
            print("开始GUI主循环")
            self.root.protocol("WM_DELETE_WINDOW", self.close)  # 绑定关闭事件
            self.root.mainloop()
            print("GUI主循环结束")
        except Exception as e:
            print(f"GUI主循环异常: {e}")
            self.logger.error(f"GUI主循环异常: {e}", exc_info=True)

    def close(self):
        """
        关闭GUI时调用，移除日志处理器
        """
        self.logger.removeHandler(self.gui_handler)
        self.root.destroy()

    def get_all_accounts(self) -> List[str]:
        """
        获取所有节点的账户地址列表
        """
        accounts = []
        for node in self.nodes:
            accounts.extend([address for address in node.accounts.keys()])
        # 去除重复的地址
        unique_accounts = list(set(accounts))
        return unique_accounts

    def get_current_node_accounts(self) -> List[str]:
        """
        获取当前选定节点的账户地址列表
        """
        selected_node = self.get_selected_node()
        if not selected_node:
            return []
        return list(selected_node.accounts.keys())

# =========================
# 主程序
# =========================

def main():
    """
    主程序，初始化区块链、启动GUI和网络节点
    """
    try:
        print("初始化日志系统")
        # 创建日志目录
        if not os.path.exists("logs"):
            os.makedirs("logs")
        # 配置全局日志
        logging.basicConfig(
            filename="logs/app.log",
            level=logging.INFO,
            format='%(asctime)s - %(name)s - %(levelname)s - %(message)s'
        )
        logger = logging.getLogger("Main")
        print("日志系统初始化完成")

        print("初始化网络")
        network = Network()
        print("网络初始化完成")

        print("初始化区块链")
        shared_blockchain = Blockchain(difficulty=2, block_reward=50.0)
        print("区块链初始化完成")

        print("启动网络节点")
        # 创建5个节点，所有节点共享同一个区块链实例
        nodes = [Node(node_id=i, blockchain=shared_blockchain, network=network) for i in range(1, 6)]
        for node in nodes:
            node.start()
            print(f"节点 {node.node_id} 已启动")
            # 打印节点账户
            print(f"节点 {node.node_id} 的账户:")
            for address in node.accounts.keys():
                print(f" - {address}")
            # 注册节点到网络
            network.register_node(node)

        print("初始化GUI")
        gui = BlockchainGUI(blockchain=shared_blockchain, logger=logger, nodes=nodes, network=network)  # GUI初始连接到第一个节点
        print("GUI初始化完成")

        print("运行GUI主循环")
        gui.run()
    except Exception as e:
        print(f"程序异常: {e}")
        logger.error(f"程序异常终止: {e}", exc_info=True)
    finally:
        print("程序结束，停止所有节点")
        for node in nodes:
            node.running = False
            node.auto_stop_event.set()
        for node in nodes:
            node.join()
        print("所有节点已停止")
        logger.info("所有网络节点已停止。")

if __name__ == "__main__":
    main()
