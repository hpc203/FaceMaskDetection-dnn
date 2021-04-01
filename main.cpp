#define _CRT_SECURE_NO_WARNINGS
#include <iostream>
#include <opencv2/dnn.hpp>
#include <opencv2/imgproc.hpp>
#include <opencv2/highgui.hpp>
#include <math.h>
using namespace cv;
using namespace dnn;
using namespace std;

class FaceMask
{
    public:
        FaceMask(const float conf_thresh = 0.5, const float iou_thresh = 0.4);
        void detect(Mat &srcimg);
    private:
        const int feature_map_sizes[5][2] = {{33, 33}, {17, 17}, {9, 9}, {5, 5}, {3, 3}};
        const float anchor_sizes[5][2] = {{0.04, 0.056}, {0.08, 0.11}, {0.16, 0.22}, {0.32, 0.45}, {0.64, 0.72}};
        const float anchor_ratios[3] = {1, 0.62, 0.42};
        const float variances[4] = {0.1, 0.1, 0.2, 0.2};
        float conf_thresh;
        float iou_thresh;
        const Size target_shape = Size(260, 260);
        const int num_prior = 5972;
        float* prior_data;
        Net net;

        void generate_priors();
        void decode(Mat loc, Mat conf, vector<Rect>& boxes, vector<float>& confidences, vector<int>& classIds, const int srcimg_h, const int srcimg_w);
};

FaceMask::FaceMask(const float conf_thresh, const float iou_thresh)
{
    this->conf_thresh = conf_thresh;
    this->iou_thresh = iou_thresh;
    this->net = readNet("models/face_mask_detection.caffemodel", "models/face_mask_detection.prototxt");
    this->generate_priors();
}

void FaceMask::generate_priors()
{
    this->prior_data = new float[this->num_prior *4];
    float* pdata = prior_data;
    int i = 0, j = 0, h = 0, w = 0;
    float height = 0, width = 0, ratio = 0;
    for (i = 0; i < 5; i++)
    {
        const int feature_map_height = this->feature_map_sizes[i][0];
        const int feature_map_width = this->feature_map_sizes[i][1];
        for (h = 0; h < feature_map_height; h++)
        {
            for (w = 0; w < feature_map_width; w++)
            {
                ratio = sqrt(this->anchor_ratios[0]);
                for(j=0;j<2;j++)
                {
                    width = this->anchor_sizes[i][j] * ratio;
                    height = this->anchor_sizes[i][j] / ratio;
//                    pdata[0] = (w + 0.5) / feature_map_width - 0.5 * width;       ///xmin
//                    pdata[1] = (h + 0.5) / feature_map_height - 0.5 * height;      ////ymin
//                    pdata[2] = (w + 0.5) / feature_map_width + 0.5 * width;       ///xmax
//                    pdata[3] = (h + 0.5) / feature_map_height + 0.5 * height;      ////ymax
                    pdata[0] = (w + 0.5) / feature_map_width;       ///center_x
                    pdata[1] = (h + 0.5) / feature_map_height;      ////center_y
                    pdata[2] = width;       ///width
                    pdata[3] = height;      ////height
                    pdata += 4;
                }

                for(j=0;j<2;j++)
                {
                    ratio = sqrt(this->anchor_ratios[j+1]);
                    width = this->anchor_sizes[i][0] * ratio;
                    height = this->anchor_sizes[i][0] / ratio;
//                    pdata[0] = (w + 0.5) / feature_map_width - 0.5 * width;       ///xmin
//                    pdata[1] = (h + 0.5) / feature_map_height - 0.5 * height;      ////ymin
//                    pdata[2] = (w + 0.5) / feature_map_width + 0.5 * width;       ///xmax
//                    pdata[3] = (h + 0.5) / feature_map_height + 0.5 * height;      ////ymax
                    pdata[0] = (w + 0.5) / feature_map_width;       ///center_x
                    pdata[1] = (h + 0.5) / feature_map_height;      ////center_y
                    pdata[2] = width;       ///width
                    pdata[3] = height;      ////height
                    pdata += 4;
                }
            }
        }
    }
}

void FaceMask::decode(Mat loc, Mat conf, vector<Rect>& boxes, vector<float>& confidences, vector<int>& classIds, const int srcimg_h, const int srcimg_w)
{
    if(loc.dims==3)
    {
        loc = loc.reshape(0, this->num_prior);
    }
    if(conf.dims==3)
    {
        conf = conf.reshape(0, this->num_prior);
    }
    float predict_xmin = 0, predict_ymin = 0, predict_w = 0, predict_h = 0;
	int srcimg_xmin = 0, srcimg_ymin = 0;
    int i = 0;
    for(i=0;i<this->num_prior;i++)
    {
        Mat scores = conf.row(i).colRange(0, 2);
        Point classIdPoint;
        double score;
        // Get the value and location of the maximum score
        minMaxLoc(scores, 0, &score, 0, &classIdPoint);
        if (score>this->conf_thresh)
        {
            const int row_ind = i * 4;
            const float* pbox = (float*)loc.data + row_ind;
            predict_w = exp(pbox[2] * this->variances[2]) * this->prior_data[row_ind + 2];
            predict_h = exp(pbox[3] * this->variances[3]) * this->prior_data[row_ind + 3];
            predict_xmin = pbox[0] * this->variances[0] * this->prior_data[row_ind + 2] + this->prior_data[row_ind] - 0.5 * predict_w;
            predict_ymin = pbox[1] * this->variances[1] * this->prior_data[row_ind + 3] + this->prior_data[row_ind + 1] - 0.5 * predict_h;
            classIds.push_back(classIdPoint.x);
            confidences.push_back(score);
			srcimg_xmin = (int)max(predict_xmin * srcimg_w, 0.f);
			srcimg_ymin = (int)max(predict_ymin * srcimg_h, 0.f);
            boxes.push_back(Rect(srcimg_xmin, srcimg_ymin, (int)(predict_w * srcimg_w), (int)(predict_h * srcimg_h)));
        }
    }
}

void FaceMask::detect(Mat &srcimg)
{
    int height = srcimg.rows;
    int width = srcimg.cols;
    Mat blob = blobFromImage(srcimg, 1/255.0, this->target_shape);
    this->net.setInput(blob);
    vector<Mat> outs;
    this->net.forward(outs, this->net.getUnconnectedOutLayersNames());
    ////post process
    vector<int> classIds;
    vector<float> confidences;
    vector<Rect> boxes;
    this->decode(outs[0], outs[1], boxes, confidences, classIds, height, width);
    vector<int> indices;
    NMSBoxes(boxes, confidences, this->conf_thresh, this->iou_thresh, indices);
    for (size_t i = 0; i < indices.size(); ++i) 
	{
        int idx = indices[i];
        Rect box = boxes[idx];
        if(classIds[idx]==1)
        {
            rectangle(srcimg, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 0, 255), 2);
            putText(srcimg, "No mask", Point(box.x, box.y -10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 0, 255), 1);
        }
        else
        {
            rectangle(srcimg, Point(box.x, box.y), Point(box.x + box.width, box.y + box.height), Scalar(0, 255, 0), 2);
            putText(srcimg, "wear mask", Point(box.x, box.y -10), FONT_HERSHEY_SIMPLEX, 0.75, Scalar(0, 255, 0), 1);
        }
    }
}

int main()
{
    FaceMask model;
	string imgpath = "img/demo2.jpg";
	Mat srcimg = imread(imgpath);
    model.detect(srcimg);

    static const string kWinName = "Deep learning object detection in OpenCV";
    namedWindow(kWinName, WINDOW_NORMAL);
    imshow(kWinName, srcimg);
    waitKey(0);
    destroyAllWindows();
}
