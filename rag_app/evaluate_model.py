from transformers import T5Tokenizer, T5ForConditionalGeneration
from sklearn.metrics import accuracy_score, classification_report, confusion_matrix
import torch
import numpy as np
from utils import calculate_similarity
import seaborn as sns
import matplotlib.pyplot as plt

MODEL_PATH = "google/flan-t5-base"

def evaluate(custom_input=False):
    print("\n=== Model Evaluation ===")
    print("Usage:")
    print("- Default mode: Evaluates the model using predefined test cases")
    print("- Custom mode: Evaluates the model with your own context, question, and answer")
    print("- Target accuracy: 70% or higher")
    print("")
    
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH).to(device)
    
    # Additional metrics
    rouge = Rouge()

    # Test cases based on the networks document
    test_cases = [
        {
            "context": "A computer network is a system of interconnected computers and devices that can communicate with each other. These networks allow devices to share resources, exchange data, and communicate over various distances.",
            "question": "What is a computer network?",
            "answer": "a system of interconnected computers and devices that can communicate with each other"
        },
        {
            "context": "Types of Computer Networks:\n\n1. Local Area Network (LAN)\n- Covers a small geographic area (typically within a building or campus)\n- High-speed connections (100 Mbps to 10 Gbps)\n- Common in offices and homes\n- Examples: Ethernet, Wi-Fi networks\n\n2. Wide Area Network (WAN)\n- Covers large geographic areas (cities, countries, or globally)\n- Connects multiple LANs\n- Used by organizations with multiple locations\n- Examples: Internet, leased lines",
            "question": "What is the difference between LAN and WAN?",
            "answer": "LAN covers a small geographic area within a building or campus with high-speed connections, while WAN covers large geographic areas connecting multiple LANs across cities, countries, or globally"
        },
        {
            "context": "Network Security:\n\n1. Common Security Measures\n- Firewalls: Block unauthorized access\n- Encryption: Protect data in transit\n- Authentication: Verify user identity\n- Antivirus: Protect against malware\n\n2. Security Protocols\n- SSL/TLS: Secure web communications\n- SSH: Secure remote access\n- HTTPS: Secure web browsing\n- WPA/WPA2: Secure wireless networks",
            "question": "What are the main security measures in computer networks?",
            "answer": "firewalls, encryption, authentication, and antivirus protection"
        },
        {
            "context": "Network Topologies:\n\n1. Star Topology\n- All devices connect to a central hub\n- Easy to manage and troubleshoot\n- Single point of failure\n\n2. Bus Topology\n- All devices connect to a single cable\n- Simple and inexpensive\n- Limited scalability\n\n3. Ring Topology\n- Devices connected in a circular loop\n- Equal data transmission\n- Complex to implement",
            "question": "What are the different types of network topologies?",
            "answer": "star topology, bus topology, and ring topology"
        },
        {
            "context": "Network Performance Metrics:\n\n1. Bandwidth: Maximum data transfer rate\n2. Latency: Time taken for data to travel\n3. Throughput: Actual data transfer rate\n4. Packet Loss: Percentage of lost data packets\n5. Jitter: Variation in packet arrival times",
            "question": "What are the key network performance metrics?",
            "answer": "bandwidth, latency, throughput, packet loss, and jitter"
        }
    ]

    if custom_input:
        print("\nEnter your custom test case:")
        try:
            context = input("Context: ")
            question = input("Question: ")
            answer = input("Answer: ")
            test_cases = [{"context": context, "question": question, "answer": answer}]
        except EOFError:
            print("\nError: This script needs to be run in an interactive environment.")
            print("Try running it directly in Python shell or IDE.")
            return

    y_true = []
    y_pred = []
    similarities = []
    
    # Token-level metrics
    true_tokens = []
    pred_tokens = []

    for case in test_cases:
        prompt = f"Context: {case['context']} Question: {case['question']}"
        inputs = tokenizer(prompt, return_tensors="pt", truncation=True, padding=True).to(device)

        outputs = model.generate(
            **inputs,
            max_length=256,  # Increased max length for more natural responses
            num_beams=8,  # Increased beams for better quality
            temperature=0.9,  # Higher temperature for more diverse responses
            top_p=0.9,  # Top-p sampling for natural language
            no_repeat_ngram_size=3,
            do_sample=True
        )
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)

        # Tokenize answers
        true_tokens.extend(case['answer'].lower().strip().split())
        pred_tokens.extend(prediction.lower().strip().split())

        y_true.append(case['answer'].lower().strip())
        y_pred.append(prediction.lower().strip())
        similarities.append(calculate_similarity(prediction, case['answer']))

        print(f"\nQuestion: {case['question']}")
        print(f"Context: {case['context']}")
        print(f"Expected Answer: {case['answer']}")
        print(f"Generated Answer: {prediction}")
        print(f"Similarity Score: {similarities[-1]:.4f}")

    # Calculate token-level metrics
    from sklearn.metrics import precision_score, recall_score, f1_score
    
    # Calculate exact match metrics
    exact_match = accuracy_score(y_true, y_pred)
    avg_similarity = np.mean(similarities)
    
    # Calculate token-level metrics using sets to handle different lengths
    true_set = set(true_tokens)
    pred_set = set(pred_tokens)
    
    # Calculate precision, recall, and F1 score
    tp = len(true_set.intersection(pred_set))
    fp = len(pred_set - true_set)
    fn = len(true_set - pred_set)
    
    precision = tp / (tp + fp) if (tp + fp) > 0 else 0
    recall = tp / (tp + fn) if (tp + fn) > 0 else 0
    f1 = 2 * (precision * recall) / (precision + recall) if (precision + recall) > 0 else 0
    
    # Calculate comprehensive metrics
    rouge_scores = []
    bleu_scores = []
    exact_match_count = 0
    natural_response_count = 0
    
    for pred, true in zip(y_pred, y_true):
        # Calculate BLEU score
        reference = [word_tokenize(true)]
        candidate = word_tokenize(pred)
        bleu_score = sentence_bleu(reference, candidate)
        bleu_scores.append(bleu_score)
        
        # Calculate BLEU score
        reference = [true.split()]
        candidate = pred.split()
        bleu_scores.append(sentence_bleu(reference, candidate))
        
        # Check for exact match
        if pred == true:
            exact_match_count += 1
            
        # Check for natural language response
        if "I found" in pred or "This means" in pred or "In other words" in pred:
            natural_response_count += 1
    
    exact_match_rate = exact_match_count / len(y_true)
    natural_response_rate = natural_response_count / len(y_true)
    
    # Calculate average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    # Calculate average BLEU score
    avg_bleu = sum(bleu_scores) / len(bleu_scores)
    
    print("\n Evaluation Results:")
    print(f"Exact Match Accuracy: {exact_match_rate:.2%}")
    print(f"Natural Response Rate: {natural_response_rate:.2%}")
    print(f"Average BLEU Score: {avg_bleu:.4f}")
    print(f"Natural Response Rate: {natural_response_rate:.2%}")
    
    # Check if we meet the 70% accuracy target
    if exact_match_rate >= 0.7:
        print("\n Target accuracy achieved!")
    else:
        print(f"\n Target accuracy not met. Current accuracy: {exact_match_rate:.2%}")
    
    # Generate visualization of similarity scores
    plt.figure(figsize=(10, 6))
    plt.hist(similarities, bins=20, alpha=0.7)
    plt.title('Similarity Scores Distribution')
    plt.xlabel('Similarity Score')
    plt.ylabel('Frequency')
    plt.axvline(x=0.7, color='r', linestyle='--', label='Target Accuracy')
    plt.legend()
    plt.savefig('similarity_scores.png')
    print("\n Similarity score distribution saved to 'similarity_scores.png'")
    
    print("\nToken-level Metrics:")
    print(f"Precision: {precision:.4f}")
    print(f"Recall: {recall:.4f}")
    print(f"F1 Score: {f1:.4f}\n")
    
    # Print classification report
    print("Classification Report:")
    print(classification_report(y_true, y_pred, zero_division=0))
    
    # Generate confusion matrix
    print("\nConfusion Matrix:")
    labels = list(set(y_true + y_pred))
    cm = confusion_matrix(y_true, y_pred, labels=labels)
    
    # Plot confusion matrix
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm, annot=True, fmt="d", xticklabels=labels, yticklabels=labels, cmap="Blues")
    plt.xlabel("Predicted")
    plt.ylabel("Actual")
    plt.title("Confusion Matrix")
    plt.tight_layout()
    plt.show()

if __name__ == "__main__":
    evaluate()
