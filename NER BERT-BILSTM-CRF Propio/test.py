    import torch
    import pickle
    from transformers import BertTokenizer, BertModel
    from torch.utils.data import Dataset, DataLoader
    import pandas as pd
    import torch.nn as nn
    import torch.optim as optim
    from sklearn.metrics import f1_score
    from TorchCRF import CRF
    from torch.nn.utils.rnn import pad_sequence

    class NERModel(nn.Module):
        def __init__(self, num_labels, hidden_dim):
            super(NERModel, self).__init__()
            self.bert = BertModel.from_pretrained('bert-base-multilingual-cased')
            self.lstm = nn.LSTM(input_size=768, hidden_size=hidden_dim, bidirectional=True, batch_first=True)
            self.hidden2label = nn.Linear(hidden_dim * 2, num_labels)
            self.crf = CRF(num_labels)
            
        def forward(self, input_ids, attention_masks):
            outputs = self.bert(input_ids, attention_mask=attention_masks)
            sequence_output = outputs.last_hidden_state
            lstm_output, _ = self.lstm(sequence_output)
            logits = self.hidden2label(lstm_output)
            return logits

    print("Carga del dataset")
    dataset = torch.utils.data.TensorDataset()
    with open('dataset_train2.pkl', 'rb') as f:
        dataset = pickle.load(f)
    print("Dataset cargado")
    # Define model parameters
    num_labels = 85  # Replace with the actual number of NER labels
    hidden_dim = 128  # Replace with the desired hidden dimension of the BiLSTM

    # Create the NER model instance
    model = NERModel(num_labels, hidden_dim)

    print("Modelo creado")

    # Define the loss function and optimizer
    loss_fn = model.crf
    optimizer = optim.AdamW(model.parameters())

    # Define the data loader
    batch_size = 32
    dataloader = DataLoader(dataset, batch_size=batch_size, shuffle=True)
    print("Dataloader creado")

    # Training loop
    num_epochs = 10  # Replace with the desired number of training epochs
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    model.to(device)

    for epoch in range(num_epochs):
        total_loss = 0
        for batch in dataloader:
            batch = [data.to(device) for data in batch]
            i_i, a_m, e = batch
                
            optimizer.zero_grad()
            logits = model(i_i, a_m)
            loss = -model.crf(logits, e)
            loss.backward()
            optimizer.step()
                
            total_loss += loss.item()
        
        print(f"Epoch {epoch+1}/{num_epochs}, Loss: {total_loss:.4f}")

    # Save the trained model
    torch.save(model.state_dict(), "ner_model_2.pt")


    # Set the mode to evaluation mode
    model.eval()

    predictions = []
    for batch in dataloader:
        batch = [data.to(device) for data in batch]
        input_ids, attention_masks, _ = batch
        with torch.no_grad():
            logits = model(input_ids, attention_masks)
            predicted_labels = model.crf.decode(logits)
        predictions.extend(predicted_labels)

     # Assuming you have the ground truth labels for evaluation
    ground_truth_labels = etiquetas

    # Flatten the predictions and ground truth labels
    flat_predictions = [label for batch in predictions for label in batch]
    flat_ground_truth = [label for labels in ground_truth_labels for label in labels]

    # Calculate the F1 score
    f1 = f1_score(flat_ground_truth, flat_predictions, average='micro')
    print("F1 score TRAIN:", f1)

    dataset_test = torch.utils.data.TensorDataset(padded_input_ids_test, attention_masks_test, etiquetas_test)
    dataloader_test = DataLoader(dataset_test, batch_size=batch_size, shuffle=True)

    predictions_test = []
    for batch in dataloader_test:
        batch = [data.to(device) for data in batch]
        input_ids_test, attention_masks_test, _ = batch
        with torch.no_grad():
            logits_test = model(input_ids_test, attention_masks_test)
            predicted_labels_test = model.crf.decode(logits_test)
        predictions_test.extend(predicted_labels_test)

     # Assuming you have the ground truth labels for evaluation
    ground_truth_labels_test = etiquetas_test

    # Flatten the predictions and ground truth labels
    flat_predictions_test = [label for batch in predictions_test for label in batch]
    flat_ground_truth_test = [label for labels in ground_truth_labels_test for label in labels]

    # Calculate the F1 score
    f1_test = f1_score(flat_ground_truth_test, flat_predictions_test, average='micro')
    print("F1 score TEST:", f1_test)






