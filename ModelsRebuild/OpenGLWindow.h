#pragma once

#define GLEW_STATIC
#include <glew.h>
#include <glfw3.h>

#include <glm.hpp>
#include <gtc\matrix_transform.hpp>

#include <opencv2\core.hpp>

#include <iostream>
#include <vector>

using namespace std;

void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode);
void InitAxis(GLuint &VAO, GLuint *VBO);
void InitGLFW();
GLuint LoadShader(const GLchar *vertexShaderCode, const GLchar *fragmentShaderCode);
void InitPoints(vector<cv::Point3f> &points, GLuint &VAO);
void DrawPoints();
int CreateOpenGLWindow(vector<cv::Point3f> &points);


