#######################################################
# One Time Truth-Table (OTTT) protocol implementation #
#######################################################

#################
# Offline phase #
#################

import random

DEALER_FILE = "dealer.txt"
ALICE_FILE = "alice.txt"
BOB_FILE = "bob.txt"


class Alice:
    def __init__(self, n, x):
        self.n = n
        self.x = x
        with open(ALICE_FILE, "w") as f:
            f.write("Alice info for OTTT protocol\n\n")
            f.write("Chosen x value: " + str(x) + "\n\n")

    def receiveDataFromDealer(self, M_A, r):
        self.M_A = M_A
        self.r = r
        with open(ALICE_FILE, "a") as f:
            f.write("Received data from dealer:\n")
            f.write("\tr = " + str(r) + "\n")
            f.write("\tM_A:")
            for row in M_A:
                if row == M_A[0]:
                    f.write(" " + str(row) + "\n")
                else:
                    f.write("\t\t " + str(row) + "\n")
            f.write("\n")

    def receiveDataFromBob(self, v, z_B):
        self.v = v
        self.z_B = z_B
        with open(ALICE_FILE, "a") as f:
            f.write("Received data from Bob:\n")
            f.write("\tv = " + str(v) + "\n")
            f.write("\tz_B = " + str(z_B) + "\n\n")

    def sendDataToBob(self, bob):
        self.u = (self.x + self.r) % (2 ** self.n)
        with open(ALICE_FILE, "a") as f:
            f.write(f"Sending data to Bob:\n")
            f.write(f"\tu = {self.u}\n\n")
        bob.receiveDataFromAlice(self.u)

    def computeOutput(self):
        output = self.M_A[self.u][self.v] ^ self.z_B
        with open(ALICE_FILE, "a") as f:
            f.write("Function output = " + str(output) + "\n\n")
            f.write("Alice finished!\n\n")


class Bob:
    def __init__(self, n, a):
        self.n = n
        self.a = a
        with open(BOB_FILE, "w") as f:
            f.write("Bob info for OTTT protocol\n\n")
            f.write("Chosen a value: " + str(a) + "\n\n")

    def receiveDataFromDealer(self, M_B, c):
        self.M_B = M_B
        self.c = c
        with open(BOB_FILE, "a") as f:
            f.write("Received data from dealer:\n")
            f.write("\tc = " + str(c) + "\n")
            f.write("\tM_B:")
            for row in M_B:
                if row == M_B[0]:
                    f.write(" " + str(row) + "\n")
                else:
                    f.write("\t\t " + str(row) + "\n")
            f.write("\n")

    def receiveDataFromAlice(self, u):
        self.u = u
        with open(BOB_FILE, "a") as f:
            f.write("Received data from Alice:\n")
            f.write("\tu = " + str(u) + "\n\n")

    def sendDataToAlice(self, alice):
        v = (self.a + self.c) % (2 ** self.n)
        z_B = self.M_B[self.u][v]
        with open(BOB_FILE, "a") as f:
            f.write("Sending data to Alice...\n")
            f.write("\tv = " + str(v) + "\n")
            f.write("\tz_B = " + str(z_B) + "\n\n")
            f.write("Bob finished!\n\n")
        alice.receiveDataFromBob(v, z_B)


class Dealer:
    def __init__(self, n):
        self.n = n
        self.truth_table = []
        with open(DEALER_FILE, "w") as f:
            f.write("Dealer info for OTTT protocol\n\n")

    def genTruthTable(self):
        for a in range(0, 2 ** self.n):
            row = []
            for x in range(0, 2 ** self.n):
                if a * x >= 4:
                    row += [1]
                else:
                    row += [0]
            self.truth_table.append(row)
        with open(DEALER_FILE, "a") as f:
            f.write("Truth table for f(a, x):\n")
            for row in self.truth_table:
                f.write("\t " + str(row) + "\n")
            f.write("\n")

    def generateRandomValues(self):
        self.r = random.randint(0, 2 ** self.n - 1)
        self.c = random.randint(0, 2 ** self.n - 1)

        with open(DEALER_FILE, "a") as f:
            f.write("Generated random values:\n")
            f.write("\tr = " + str(self.r) + "\n")
            f.write("\tc = " + str(self.c) + "\n\n")

    def generateMatrices(self):
        # shift the truth table by r rows and c columns
        shifted_tt = []
        for i in range(len(self.truth_table)):
            row = []
            for j in range(len(self.truth_table[0])):
                row += [self.truth_table[(i - self.r) % (2 ** self.n)][(j - self.c) % (2 ** self.n)]]
            shifted_tt.append(row)

        # generate a random truth table
        self.M_B = []
        for i in range(len(self.truth_table)):
            row = []
            for j in range(len(self.truth_table[0])):
                row += [random.randint(0, 1)]
            self.M_B.append(row)

        # 4. generate M_A = M_B XOR shifted truth table
        self.M_A = []
        for i in range(len(self.truth_table)):
            row = []
            for j in range(len(self.truth_table[0])):
                row += [self.M_B[i][j] ^ shifted_tt[i][j]]
            self.M_A.append(row)

        with open(DEALER_FILE, "a") as f:
            f.write("Generated matrices:\n")
            f.write("\tM_A:")
            for row in self.M_A:
                if row == self.M_A[0]:
                    f.write(" " + str(row) + "\n")
                else:
                    f.write("\t\t " + str(row) + "\n")
            f.write("\n\tM_B:")
            for row in self.M_B:
                if row == self.M_B[0]:
                    f.write(" " + str(row) + "\n")
                else:
                    f.write("\t\t " + str(row) + "\n")
            f.write("\n")

    def sendDataToAliceAndBob(self, alice, bob):
        alice.receiveDataFromDealer(self.M_A, self.r)
        bob.receiveDataFromDealer(self.M_B, self.c)
        with open(DEALER_FILE, "a") as f:
            f.write("Sending data to Alice and Bob...\n\n")
            f.write("Dealer finished!\n\n")


if __name__ == "__main__":
    n = 2
    dealer = Dealer(n)
    alice = Alice(n, random.randint(0, 3))
    bob = Bob(n, random.randint(0, 3))

    # 1. generate truth table for the function f(a, x)
    dealer.genTruthTable()

    # 2. generate random values r and c
    dealer.generateRandomValues()

    # 3. generate M_A and M_B #
    dealer.generateMatrices()

    # 4. send M_A and M_B to Alice and Bob
    dealer.sendDataToAliceAndBob(alice, bob)

    ################
    # Online phase #
    ################

    # 5. Alice sends u to Bob
    alice.sendDataToBob(bob)

    # 6. Bob sends v and z_B to Alice
    bob.sendDataToAlice(alice)

    # 7. Alice computes the output of f(a, x)
    alice.computeOutput()