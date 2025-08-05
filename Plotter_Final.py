import re
import matplotlib.pyplot as plt

# Load log file
log_file = 'loss_log.txt'
with open(log_file, 'r') as f:
    log_lines = f.readlines()

# Initialize lists
train_loss = []
val_loss = []
epochs = []

# Regular expression to extract losses
pattern = re.compile(r"Epoch (\d+).*\| TrainLoss: ([0-9.]+) \| ValLoss: ([0-9.]+)")

# Parse log file
for line in log_lines:
    match = pattern.search(line)
    if match:
        epoch = int(match.group(1))
        train = float(match.group(2))
        val = float(match.group(3))
        epochs.append(epoch)
        train_loss.append(train)
        val_loss.append(val)

# Plotting
plt.figure(figsize=(10, 6))
plt.plot(epochs, train_loss, label='Train Loss', linewidth=2)
plt.plot(epochs, val_loss, label='Validation Loss', linewidth=2)
plt.xlabel('Epoch')
plt.ylabel('Loss')
plt.title('Training and Validation Loss over Epochs')
plt.legend()
plt.grid(True)
plt.tight_layout()
plt.show()
