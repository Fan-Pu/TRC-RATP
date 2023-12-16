class CircularQueue:
    def __init__(self, capacity):
        self.capacity = capacity
        self.queue = [None] * capacity
        self.head = 0
        self.tail = 0
        self.size = 0

    def enqueue(self, value):
        if self.is_full():
            raise Exception("Queue is full")
        self.queue[self.tail] = value
        self.tail = (self.tail + 1) % self.capacity
        self.size += 1

    def dequeue(self):
        if self.is_empty():
            raise Exception("Queue is empty")
        value = self.queue[self.head]
        self.queue[self.head] = None
        self.head = (self.head + 1) % self.capacity
        self.size -= 1
        return value

    def is_empty(self):
        return self.size == 0

    def is_full(self):
        return self.size == self.capacity

    def __str__(self):
        if self.is_empty():
            return "Empty Queue"
        start = self.head
        end = self.tail
        if end <= start:
            end += self.capacity
        return ' -> '.join(str(self.queue[i % self.capacity]) for i in range(start, end))

    def get_front(self):
        if self.is_empty():
            raise Exception("Queue is empty")
        return self.queue[self.head]
