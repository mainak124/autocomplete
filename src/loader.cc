#include "tensorflow/core/public/session.h"
#include "tensorflow/core/platform/env.h"

#include "tensorflow/cc/saved_model/loader.h"

using namespace tensorflow;

int LoadSessionFromSavedModel(const std::string export_dir, SavedModelBundle& bundle) {
    SessionOptions session_options;
    RunOptions run_options;

    const std::unordered_set<std::string> tags = {"serve"};
    
    Status status = LoadSavedModel(session_options,               
                            run_options, export_dir,
                            tags, &bundle);

    // return Status::OK();

    if (!status.ok()) {
        std::cout << status.ToString() << std::endl;
        return 1;
    }

    return 0;
}

void PrepareInput(const std::vector<std::string>& input_texts, std::vector<std::pair<string, tensorflow::Tensor>>& inputs) {
    Tensor text_till_t(DT_STRING, TensorShape({2}));
    Tensor text_at_t(DT_STRING, TensorShape({2}));
    Tensor src_sequence_length(DT_INT32, TensorShape({2}));

    std::vector<std::string>::const_iterator it;
    int ptr;
    std::string word;

    for (it=input_texts.begin(), ptr = 0; it != input_texts.end(); it++, ptr++) {
        word = *it;
        text_till_t.vec<string>()(ptr) = word.substr(0, word.size()-1);
        text_at_t.vec<string>()(ptr) = word.substr(word.size()-1, 1);
        src_sequence_length.vec<int>()(ptr) = word.size();
    }   

    inputs = {
        { "text_till_t", text_till_t },
        { "text_at_t", text_at_t },
        { "src_sequence_length", src_sequence_length },
    };
}

int Predict(const SavedModelBundle& bundle, const std::vector<std::pair<string, tensorflow::Tensor>>& inputs, std::vector<tensorflow::Tensor>& outputs) {
    // Run the session, evaluating our "predictions" operation from the graph
    Status status = bundle.session->Run(inputs, {"predictions:0"}, {}, &outputs);
    if (!status.ok()) {
        std::cout << status.ToString() << "\n";
        return 1;
    }

    auto output_c = outputs[0].tensor<string, 3>();
    std::vector<string> out_predicted;

    TensorShape output_shape = outputs[0].shape();
    int num_data = output_shape.dim_size(0);
    int num_beams = output_shape.dim_size(2);
    int num_chars = output_shape.dim_size(1);

    for (int batch_id=0; batch_id<num_data; batch_id++) {
        for (int beam_id=0; beam_id<num_beams; beam_id++) {
            string word = "";
            for (int char_id=0; char_id<num_beams; char_id++) {
                word += output_c(batch_id, char_id, beam_id);
            }
            std::cout << word << std::endl;
        }
        std::cout << std::endl; 
    }

    return 0;
}

int main(int argc, char* argv[]) {

    SavedModelBundle bundle;
    const std::string export_dir = "../saved_model/";
    LoadSessionFromSavedModel(export_dir, bundle);

    std::vector<std::string> input_strings({"ff", "av"});

    // declare input and output containers
    std::vector<std::pair<string, tensorflow::Tensor>> inputs;
    std::vector<tensorflow::Tensor> outputs;

    PrepareInput(input_strings, inputs);

    Predict(bundle, inputs, outputs);

    // close the session
    Status status = bundle.session->Close();
    if (!status.ok()) {
        std::cerr << "error while closing session" << std::endl;
    }
    // delete bundle.session;

    return 0;

}
