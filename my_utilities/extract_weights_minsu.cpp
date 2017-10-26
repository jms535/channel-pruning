#include <cstring>
#include <cstdlib>
#include <vector>
#include <string>
#include <iostream>
#include <stdio.h>
#include <assert.h>

#include "caffe/caffe.hpp"
#include "caffe/util/io.hpp"
#include "caffe/blob.hpp"

using namespace caffe;
using namespace std;

#define N 9 // the number of label types
#define Nf 4 // the number of print features

int main(int argc, char** argv)
{
    string str1 = argv[1];
    string str2 = argv[2];
    int i = 0;
    int k = 0;
    Caffe::set_mode(Caffe::GPU);

    // Set to TEST Phase
    //Caffe::set_phase(Caffe::TEST);

    // Load net
    // Assume you are in Caffe master directory
    Net<float> net("./" + str1, TEST);
    // Load pre-trained net (binary proto)
    // Assume you are already trained the bm example.
    net.CopyTrainedLayersFrom("./" + str2);

    vector< shared_ptr< Layer< float > > >  layers = net.layers();
    int tot_layer_num = (int)layers.size();
    cout << "layer num = " << tot_layer_num << endl;
    
    for(i=0; i<tot_layer_num; i++){
        string type = layers[i]->type();
        if(type.compare("Convolution")!=0 && type.compare("InnerProduct")!=0)
            continue;
        cout << "layer#" << i << " : weight = " << endl;
        cout << "   ";
        for(k=0; k<layers[i]->blobs()[0]->count(); k++){
            cout << " " << layers[i]->blobs()[0]->cpu_data()[k];
        }
        cout << endl;

        cout << "layer" << i << ": bias = " << endl;
        cout << "   ";
        for(k=0; k<layers[i]->blobs()[1]->count(); k++){
            cout << " " << layers[i]->blobs()[1]->cpu_data()[k];
        }
        cout << endl;

    }
    
     
/* 
    
    float loss = 0.0;
    vector<Blob<float>*> results = net.ForwardPrefilled(&loss);
    LOG(INFO) << "Result size: "<< results.size();

    // Log how many blobs were loaded
    LOG(INFO) << "Blob size: "<< net.input_blobs().size();

    LOG(INFO)<< "-------------";
    LOG(INFO)<< " prediction ";

    // Get probabilities
    const boost::shared_ptr<Blob<float> >& probLayer = net.blob_by_name("prob");
    const float* probs_out = probLayer->cpu_data();
    
    // Get feature
    const boost::shared_ptr<Blob<float> >& featLayer = net.blob_by_name("ip1");
    

    // Get argmax results
    const boost::shared_ptr<Blob<float> >& argmaxLayer = net.blob_by_name("argmax");

    // predict
    const boost::shared_ptr<Blob<float> >& labelBlob = net.blob_by_name("label");
    const float *label_data = labelBlob->cpu_data();
    int correct = 0;
    int wrong = 0;
    int cnt = 0;
    int tp = 0;
    int tn = 0;
    int fp = 0;
    int fn = 0;
    float threshold = 0.001;
    int thcorrect = 0;
    int thwrong = 0;
    int thtp = 0;
    int thtn = 0;
    int thfp = 0;
    int thfn = 0;

    int behtp[N] = {0};
    int behtn[N] = {0};
    int behfp[N] = {0};
    int behfn[N] = {0};

    cout << labelBlob->count() << endl;
    cout << "threshold: " << threshold << endl;
    cout << "==================" << endl;

    for(int i=0; i<labelBlob->count(); i++){
        int truelabel = label_data[i];
        int predictedlabel = 0;
        float* score = new float[N];
        float* feat = new float[Nf];
        cnt++;

        // features
        for(int j=0; j<Nf; j++){
            feat[j] = featLayer->data_at(i,j,0,0);
        }

        //cout << truelabel << " : ";
        for(int j=0; j<N; j++){
            score[j] = probLayer->data_at(i,j,0,0);
            //cout << score[j] << ", ";
        }

        for(int j=0; j<N; j++){
            if(score[j] > score[predictedlabel]){
                predictedlabel = j;
            }
        }
        //cout << ": " << predictedlabel << " : ";// << endl;
/*
        if(truelabel == predictedlabel){
            correct++;

            if(truelabel==0){
                tn++;
            }
            else{
                tp++;
            }
        }
        else{
            if(predictedlabel!=0){
                fp++;
            }
        }

        if(predictedlabel != 0){
            cout << truelabel << " : " << predictedlabel << " : " << score[predictedlabel] << endl;
            
        }
        else{ // if predicted == 0
            if(truelabel != 0){
                fn++;
            }
        }
*/
        // eval 1
        //if(predictedlabel != 0){
            //cout << truelabel << " : " << predictedlabel << " : " << score[predictedlabel] << endl;   
        //}
/*
        cout << truelabel << " : " << predictedlabel << " : ";

        // print hidden features
        for(int j=0; j<Nf; j++){
            cout << feat[j] << ", ";
        }
        cout << endl;

        //print prob
        //for(int j=0; j<15; j++){
        //    cout << score[j] << ", ";
        //}
        //cout << endl;
        

        if(truelabel == predictedlabel){
            if(truelabel==0){
                tn++;
            }
            else{
                correct++;
            }
        }
        else{
            if(truelabel==0){
                fp++;
            }
            else if(predictedlabel==0){
                fn++;
            }
            else{
                wrong++;
            }
        }

        // for(int j=0; j<66; j++){
        //  cout << dataBlob->data_at(i,0,0,j) << ", ";
        // }
        //cout<<endl;

        // eval2. final predict analysis
        if(score[predictedlabel] > threshold){ // if event occur predicted
            if(truelabel==predictedlabel){
                if(truelabel==0){
                    thtn++;
                }
                else{
                    thcorrect++;
                }
            }
            else{
                if(truelabel==0){
                    thfp++;
                }
                else if(predictedlabel==0){
                    thfn++;
                }
                else{
                    thwrong++;
                }
            }
        }
        else{
            if(truelabel==0){
                thtn++;
            }
            else{
                thfn++;
            }
        }

        // eval3. each F-measure with applying threshold
        for(int j=1; j<N; j++){
            if(score[predictedlabel] > threshold){
                if(truelabel==j && predictedlabel==j){
                    behtp[j]++;
                    //if(truelabel==3){
                    //    cout<< "hihihi : " << i << endl;
                    //}
                }
                else if(truelabel==0 && predictedlabel==0){
                    behtn[j]++;
                }
                else if(truelabel!=j && predictedlabel==j){
                    behfp[j]++;
                }
                else if(truelabel==j && predictedlabel!=j){
                    behfn[j]++;
                }
                else{
                    behtn[j]++;
                }
            }
            else{
                if(truelabel==j){
                    behfn[j]++;
                }
                else{
                    behtn[j]++;
                }
            }
        }


    }
    tp = correct + wrong;
    thtp = thcorrect + thwrong;

    cout << "====================" << endl;
    cout << "total acc = " << (correct+tn) << " / " << cnt << " = " << (((double)(correct+tn))/(double)cnt) << endl;
    cout << "event correct ratio = " << correct << " / " << (correct + wrong + fp+ fn) << " = " << ((double)correct/(double)((correct + wrong + fp+ fn))) << endl;
    cout << "precision = " << correct << " / (" << tp << " + " << fp << ") = " << ((double)correct/((double)tp + (double)fp)) << endl;
    cout << "recall = " << correct << " / (" << tp << " + " << fn << ") = " << ((double)correct/((double)tp + (double)fn)) << endl;

    cout << "correct=" << correct << ", wrong=" << wrong << endl;
    cout << "tp=" << tp << ", fp=" << fp << endl;
    cout << "fn=" << fn << ", tn=" << tn << endl;

    cout << "========== final prediction ==========" << endl;
    cout << "total event correct ratio = " << thcorrect << " / " << (thcorrect + thwrong + thfp + thfn) << " = " << ((double)thcorrect/(double)(thcorrect + thwrong + thfp + thfn)) << endl;
    cout << "correct ratio without FN = " << thcorrect << " / " << (thcorrect + thwrong + thfp) << " = " << ((double)thcorrect/((double)(thcorrect + thwrong + thfp))) << endl;

    cout << "correct=" << thcorrect << ", wrong=" << thwrong << endl;
    cout << "tp=" << thtp << ", fp=" << thfp << endl;
    cout << "fn=" << thfn << ", tn=" << thtn << endl;

    for(int i=1; i<N; i++){
        cout << "========== maneuver " << i << " ==========" << endl;
        cout << "F-measure = " << ((double)(2*behtp[i]) / (double)(2*behtp[i] + behfp[i] + behfn[i])) << endl;
        cout << "precision = " << ((double)behtp[i] / (double)(behtp[i] + behfp[i])) << endl;
        cout << "recall = " << ((double)behtp[i] / (double)(behtp[i] + behfn[i])) << endl;

        cout << "tp=" << behtp[i] << ", fp=" << behfp[i] << endl;
        cout << "fn=" << behfn[i] << ", tn=" << behtn[i] << endl;
    }
    
    // cout << probLayer->data_at(0,0,0,0) << endl;
    // cout << probLayer->data_at(0,1,0,0) << endl;
    // cout << probLayer->data_at(0,14,0,0) << endl;


    // Display results
    LOG(INFO) << "---------------------------------------------------------------";
    const float* argmaxs = argmaxLayer->cpu_data();
    for (int i = 0; i < argmaxLayer->num(); i++) 
    {
        LOG(INFO) << "Pattern:"<< i << " class:" << argmaxs[i*argmaxLayer->height() + 0] << " Prob=" << probs_out[i*probLayer->height() + 0];
    }
    LOG(INFO)<< "-------------";

*/
    return 0;
}
