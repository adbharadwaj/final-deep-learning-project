from CustomTextCNN import CustomTextCNN
from utils import *
from sklearn.model_selection import KFold

# Load data
x_text, y = load_data_and_labels()

# Build vocabulary
max_document_length = max([len(x.split(" ")) for x in x_text])
vocab_processor = learn.preprocessing.VocabularyProcessor(max_document_length)
x = np.array(list(vocab_processor.fit_transform(x_text)))
print("Vocabulary Size: {:d}".format(len(vocab_processor.vocabulary_)))


encodedPathwayA, encodedPathwayB = list(vocab_processor.transform(['pathwayA pathwayB']))[0][:2]
print("encodedPathwayA = %s" % encodedPathwayA, "encodedPathwayB = %s" % encodedPathwayB)

word_distancesA = load_word_distancesA()
word_distancesB = load_word_distancesB()

pos_embedding = load_pos_embedding()


def perform_cross_validation(experiment="default", 
                             embedding_size=128, filter_sizes=[3,4,5], 
                             num_filters=128, batch_size=64, 
                             l2_reg_lambda=0.0, num_epochs=20,
                             include_word_embedding=True,
                             include_position_embedding=True,
                             include_pos_embedding=True):
    
    # Creating folds
    kf = KFold(n_splits=4, random_state=5, shuffle=True)
    for k, (train_index, test_index) in enumerate(kf.split(x, y)):
        x_train, x_dev = x[train_index], x[test_index]
        y_train, y_dev = y[train_index], y[test_index]

        train_word_distancesA = word_distancesA[train_index]
        train_word_distancesB = word_distancesB[train_index]

        test_word_distancesA = word_distancesA[test_index]
        test_word_distancesB = word_distancesB[test_index]

        train_pos_embedding = pos_embedding[train_index]
        test_pos_embedding = pos_embedding[test_index]

        print("Starting Fold: %s =>" % k, "Train/Dev split: {:d}/{:d}".format(len(y_train), len(y_dev)))

        model = CustomTextCNN(sequence_length=x_train.shape[1],
                                        vocab_processor=vocab_processor, 
                                          num_epochs=num_epochs, 
                                          embedding_size=embedding_size,
                                          filter_sizes=filter_sizes, 
                                          num_filters=num_filters,
                                          batch_size=batch_size, 
                                          l2_reg_lambda=l2_reg_lambda,
                                          evaluate_every=300, 
                                          results_dir=experiment+'_fold_%s'%k, 
                                          word_embedding=include_word_embedding,
                                          position_embedding=include_position_embedding,
                                          pos_embedding=include_pos_embedding)
        
        model.train_network(x_train, y_train, x_dev, y_dev, 
                            train_word_distancesA, train_word_distancesB, test_word_distancesA, test_word_distancesB,
                           train_pos_embedding, test_pos_embedding)
                       
                       
print("Varying number of filter")
filter_numbers = [128, 256]
for i in range(len(filter_numbers)):
    experiment = "num_filter_%s" % filter_numbers[i]
    print("Starting Experiment - %s \n\n\n" % experiment)
    perform_cross_validation(experiment=experiment, 
                             num_filters=filter_numbers[i])
