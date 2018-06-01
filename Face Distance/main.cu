#include <iostream>
#include <vector>
#include <string>
#include <opencv2/core/core.hpp>
#include <opencv2/highgui/highgui.hpp>
#include <opencv2/imgproc/imgproc.hpp>
#include <opencv2/core/cuda.hpp>
#include <opencv2/opencv.hpp>
#include <opencv2/objdetect.hpp>
#define AVERAGE_DISTANCE 294.0

bool useGpu = true;
bool useCamera = true;
std::string image_name;

using std::cout;
using std::endl;
float averageDistance = 114;

void start_gpu(cv::VideoCapture &camera)
{
	cout << "Number of cuda capable devices: " << cv::cuda::getCudaEnabledDeviceCount() << endl;

	cv::Ptr<cv::cuda::CascadeClassifier> cascade_gpu;
	cv::Ptr<cv::cuda::CascadeClassifier> cascade_eye_gpu;
	cv::cuda::GpuMat input_gpu;
	cv::cuda::GpuMat output_gpu;
	cv::cuda::GpuMat faces_gpu;
	cv::cuda::GpuMat eyes_gpu;

	cv::Mat image = cv::imread(image_name);
	cv::Mat input;
	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eyes;
	float faceDistance;

	// load cascade face for cuda
	cascade_gpu = cv::cuda::CascadeClassifier::create("haarcascade_frontalface_default_cuda.xml");
	cascade_eye_gpu = cv::cuda::CascadeClassifier::create("haarcascade_eye_cuda.xml");

	cout << "USING CUDA" << endl;
	while (true)
	{
		int64 startTime = cv::getTickCount();
		if (useCamera) {
			camera >> input;
		}
		else {
			input = image.clone();
		}

		input_gpu.upload(input);
		cv::cuda::cvtColor(input_gpu, output_gpu, CV_BGR2GRAY);
		cascade_gpu->setFindLargestObject(true);
		cascade_gpu->setScaleFactor(1.1);
		cascade_gpu->setMinNeighbors(4);
		cascade_gpu->setMinObjectSize(cv::Size(50, 50));
		cascade_gpu->detectMultiScale(output_gpu, faces_gpu);
		cascade_gpu->convert(faces_gpu, faces);
		if (faces.size() > 0) {
			cv::rectangle(input, cv::Point(faces[0].x, faces[0].y), cv::Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height), cv::Scalar::all(255));

			// detect eyes
			cv::cuda::GpuMat roi = output_gpu(faces[0]);
			std::vector<cv::Point> eyesCenter;

			cascade_eye_gpu->detectMultiScale(roi, eyes_gpu);
			cascade_gpu->setFindLargestObject(true);
			cascade_gpu->setScaleFactor(1.1);
			cascade_gpu->setMinNeighbors(100);
			cascade_gpu->setMinObjectSize(cv::Size(50, 50));
			cascade_gpu->convert(eyes_gpu, eyes);
			for (size_t j = 0; j < eyes.size(); j++) {
				cv::rectangle(input, cv::Point(faces[0].x + eyes[j].x, faces[0].y + eyes[j].y), cv::Point(faces[0].x + eyes[j].x + eyes[j].width, faces[0].y + eyes[j].y + eyes[j].height), cv::Scalar(0, 0, 255));
				eyesCenter.push_back(cv::Point(faces[0].x + eyes[j].x + eyes[j].width / 2, faces[0].y + eyes[j].y + eyes[j].height / 2));
			}

			if (eyesCenter.size() >= 2) {
				cv::line(input, cv::Point(eyesCenter[0].x, eyesCenter[0].y), cv::Point(eyesCenter[1].x, eyesCenter[1].y), (0, 0, 255), 2);
				for (size_t j = 0; j < eyesCenter.size(); j++) {
					cv::circle(input, cv::Point(eyesCenter[j].x, eyesCenter[j].y), 2, cv::Scalar(0, 255, 0));
				}

				double eyeDistance = cv::norm(cv::Point(eyesCenter[0].x, eyesCenter[0].y) - cv::Point(eyesCenter[1].x, eyesCenter[1].y));
				faceDistance = AVERAGE_DISTANCE * (averageDistance / eyeDistance);
			}
		}

		std::string faceDistanceText(16, '\0');
		auto faceWritten = std::snprintf(&faceDistanceText[0], faceDistanceText.size(), "%.2f", faceDistance / 10);
		faceDistanceText.resize(faceWritten);
		cv::putText(input, "Face distance: " + faceDistanceText + " cm", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

		double fps = cv::getTickFrequency() / (cv::getTickCount() - startTime);

		std::string s(16, '\0');
		auto written = std::snprintf(&s[0], s.size(), "%.2f", fps);
		s.resize(written);
		putText(input, "FPS " + s, cv::Point(10.0, 60.0), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

		cv::imshow("Face Distance", input);

		std::cout << "FPS : " << fps << std::endl;
		if (cv::waitKey(1) == 27) break;
	}

	faces_gpu.release();
	eyes_gpu.release();
	input_gpu.release();
}

void start_cpu(cv::VideoCapture &camera)
{
	cv::CascadeClassifier faceCascade;
	cv::CascadeClassifier eyesCascade;

	cv::Mat image = cv::imread(image_name);
	cv::Mat input;
	cv::Mat gray;

	std::vector<cv::Rect> faces;
	std::vector<cv::Rect> eyes;
	float faceDistance;

	// load cascade face for cpu
	faceCascade.load("haarcascade_frontalface_default_cpu.xml");
	eyesCascade.load("haarcascade_eye_cpu.xml");

	cout << "USING CPU" << endl;
	while (true)
	{
		int64 startTime = cv::getTickCount();
		if (useCamera) {
			camera >> input;
		}
		else {
			input = image.clone();
		}

		cv::cvtColor(input, gray, CV_BGR2GRAY);
		faceCascade.detectMultiScale(gray, faces, 1.1, 4, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(50, 50));
		if (faces.size() > 0) {
			cv::rectangle(input, cv::Point(faces[0].x, faces[0].y), cv::Point(faces[0].x + faces[0].width, faces[0].y + faces[0].height), cv::Scalar::all(255));

			// detect eyes
			cv::Mat faceROI = gray(faces[0]);
			std::vector<cv::Point> eyesCenter;
			
			eyesCascade.detectMultiScale(faceROI, eyes, 1.1, 5, 0 | cv::CASCADE_SCALE_IMAGE, cv::Size(30, 30));
			for (size_t j = 0; j < eyes.size(); j++) { 
				cv::rectangle(input, cv::Point(faces[0].x + eyes[j].x, faces[0].y + eyes[j].y), cv::Point(faces[0].x + eyes[j].x + eyes[j].width, faces[0].y + eyes[j].y + eyes[j].height), cv::Scalar(0, 0, 255));
				eyesCenter.push_back(cv::Point(faces[0].x + eyes[j].x + eyes[j].width / 2, faces[0].y + eyes[j].y + eyes[j].height / 2));
			}

			if (eyesCenter.size() >= 2) {
				cv::line(input, cv::Point(eyesCenter[0].x, eyesCenter[0].y), cv::Point(eyesCenter[1].x, eyesCenter[1].y), (0, 0, 255), 2);
				for (size_t j = 0; j < eyesCenter.size(); j++) {
					cv::circle(input, cv::Point(eyesCenter[j].x, eyesCenter[j].y), 2, cv::Scalar(0, 255, 0));
				}

				double eyeDistance = cv::norm(cv::Point(eyesCenter[0].x, eyesCenter[0].y) - cv::Point(eyesCenter[1].x, eyesCenter[1].y));
				faceDistance = AVERAGE_DISTANCE * (averageDistance / eyeDistance);
			}

			double eyeDistance = cv::norm(cv::Point(eyesCenter[0].x, eyesCenter[0].y) - cv::Point(eyesCenter[1].x, eyesCenter[1].y));
			faceDistance = AVERAGE_DISTANCE * (averageDistance / eyeDistance);
		}

		std::string faceDistanceText(16, '\0');
		auto faceWritten = std::snprintf(&faceDistanceText[0], faceDistanceText.size(), "%.2f", faceDistance / 10);
		faceDistanceText.resize(faceWritten);
		cv::putText(input, "Face distance: " + faceDistanceText + " cm", cv::Point(10, 30), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

		double fps = cv::getTickFrequency() / (cv::getTickCount() - startTime);

		std::string s(16, '\0');
		auto written = std::snprintf(&s[0], s.size(), "%.2f", fps);
		s.resize(written);
		putText(input, "FPS " + s, cv::Point(10.0, 60.0), cv::FONT_HERSHEY_SIMPLEX, 1.0, cv::Scalar(0, 255, 0), 2);

		cv::imshow("Face Distance", input);

		std::cout << "FPS : " << fps << std::endl;

		if (cv::waitKey(1) == 27) break;
	}
}

int main(int argc, char* argv[])
{
	if (argc >= 2) {
		useGpu = strcmp(argv[1], "gpu") == 0;
		cout << "Using gpu = " << useGpu << endl;

		if (argc == 3) {
			useCamera = false;
			image_name = argv[2];
			cout << "Using image = " << argv[2] << endl;
		}
	}

	cv::VideoCapture camera(0);
	cv::Mat          frame;
	if (!camera.isOpened())
		return -1;

	cv::namedWindow("Face Distance");

	if (!useGpu)
		start_cpu(camera);
	else
		start_gpu(camera);
	
	return 0;
}