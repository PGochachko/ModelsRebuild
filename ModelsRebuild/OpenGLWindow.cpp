#include "stdafx.h"
#include "OpenGLWindow.h"

glm::mat4 MVP;
glm::mat4 projection;
glm::mat4 view;
glm::mat4 model;

const GLchar* vertexShaderSource = "#version 330 core\n"
"layout (location = 0) in vec3 position;\n"
"layout (location = 1) in vec3 color;\n"
"out vec3 ourColor;\n"
"uniform mat4 MVP;\n"
"void main()\n"
"{\n"
"vec4 model = vec4(position.x, position.y, position.z, 1.0);\n"
"gl_Position = MVP*model;\n"
"ourColor = color;\n"
"}\n\0";

const GLchar* fragmentShaderSource = "#version 330 core\n"
"in vec3 ourColor;\n"
"out vec4 color;\n"
"void main()\n"
"{\n"
"color = vec4(ourColor.r, ourColor.g, ourColor.b, 1.0f);\n"
"}\n\0";


void key_callback(GLFWwindow *window, int key, int scancode, int action, int mode)
{
	if (key == GLFW_KEY_ESCAPE && action == GLFW_PRESS)
		glfwSetWindowShouldClose(window, GL_TRUE);


	if ((action == GLFW_REPEAT || action == GLFW_PRESS) && mode == 0)
	{
		switch (key)
		{
		case GLFW_KEY_W:
			MVP = glm::translate(MVP, glm::vec3(0.0f, 0.2f, 0.0f));
			break;
		case GLFW_KEY_S:
			MVP = glm::translate(MVP, glm::vec3(0.0f, -0.2f, 0.0f));
			break;
		case GLFW_KEY_A:
			MVP = glm::translate(MVP, glm::vec3(-0.2f, 0.0f, 0.0f));
			break;
		case GLFW_KEY_D:
			MVP = glm::translate(MVP, glm::vec3(0.2f, 0.0f, 0.0f));
			break;
		case GLFW_KEY_R:
			MVP = glm::translate(MVP, glm::vec3(0.0f, 0.0f, 0.2f));
			break;
		case GLFW_KEY_F:
			MVP = glm::translate(MVP, glm::vec3(0.0f, 0.0f, -0.2f));
			break;
		}
	}

	if ((action == GLFW_REPEAT || action == GLFW_PRESS) && mode == GLFW_MOD_CONTROL)
	{
		switch (key)
		{
		case GLFW_KEY_W:
			MVP = glm::rotate(MVP, 0.05f, glm::vec3(0.0f, 1.0f, 0.0f));
			break;
		case GLFW_KEY_S:
			MVP = glm::rotate(MVP, -0.05f, glm::vec3(0.0f, 1.0f, 0.0f));
			break;
		case GLFW_KEY_A:
			MVP = glm::rotate(MVP, -0.05f, glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case GLFW_KEY_D:
			MVP = glm::rotate(MVP, 0.05f, glm::vec3(1.0f, 0.0f, 0.0f));
			break;
		case GLFW_KEY_R:
			MVP = glm::rotate(MVP, 0.05f, glm::vec3(0.0f, 0.0f, 1.0f));
			break;
		case GLFW_KEY_F:
			MVP = glm::rotate(MVP, -0.05f, glm::vec3(0.0f, 0.0f, 1.0f));
			break;
		}
	}
}

void InitAxis(GLuint &VAO, GLuint *VBO)
{
	GLfloat axis[] = {-0.9f, 0.0f, 0.0f,			//x0
					   0.9f, 0.0f, 0.0f,			//x1
					   0.0f, -0.9f, 0.0f,			//y0
					   0.0f,  0.9f, 0.0f,			//y1
					   0.0f, 0.0f, -0.9f,			//z0
					   0.0f, 0.0f,  0.9f};			//z1
						

	GLfloat colorAxis[] = { 1.0f, 0.0f, 0.0f,			//x0
							1.0f, 0.0f, 0.0f,			//x1
							0.0f, 1.0f, 0.0f,			//y0
							0.0f, 1.0f, 0.0f,			//y1
							0.0f, 0.0f, 1.0f,			//z0
							0.0f, 0.0f, 1.0f };			//z1


	glGenBuffers(2, VBO);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(axis), &axis[0], GL_STATIC_DRAW);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorAxis), &colorAxis[0], GL_STATIC_DRAW);

	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glEnableVertexAttribArray(0);
	glEnableVertexAttribArray(1);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);
}

void InitGLFW()
{
	glfwInit();
	glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3);
	glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3);
	glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE);
	glfwWindowHint(GLFW_RESIZABLE, GL_FALSE);
}

GLuint LoadShader(const GLchar *vertexShaderCode, const GLchar *fragmentShaderCode)
{
	GLuint vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (vertexShader == 0)
	{
		std::cout << "Error creating vertex shader!" << std::endl;
		exit(EXIT_FAILURE);
	}
	glShaderSource(vertexShader, 1, &vertexShaderSource, NULL);
	glCompileShader(vertexShader);

	GLint success;
	glGetShaderiv(vertexShader, GL_COMPILE_STATUS, &success);
	if (success == GL_FALSE)
	{
		std::cout << "Vertex shader compilation failed!" << std::endl;
		GLint logLen;
		glGetShaderiv(vertexShader, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0)
		{
			char *log = new char[logLen];
			GLsizei written;
			glGetShaderInfoLog(vertexShader, logLen, &written, log);
			std::cout << "Shader log: " << std::endl;
			std::cout << log << std::endl;
			delete[] log;
		}
	}

	GLuint fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (fragmentShader == 0)
	{
		std::cout << "Error creating fragment shader!" << std::endl;
		exit(EXIT_FAILURE);
	}
	glShaderSource(fragmentShader, 1, &fragmentShaderSource, NULL);
	glCompileShader(fragmentShader);

	glGetShaderiv(fragmentShader, GL_COMPILE_STATUS, &success);
	if (success == GL_FALSE)
	{
		std::cout << "Fragment shader compilation failed!" << std::endl;
		GLint logLen;
		glGetShaderiv(fragmentShader, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0)
		{
			char *log = new char[logLen];
			GLsizei written;
			glGetShaderInfoLog(fragmentShader, logLen, &written, log);
			std::cout << "Shader log: " << std::endl;
			std::cout << log << std::endl;
			delete[] log;
		}
	}

	GLuint shaderProgram = glCreateProgram();
	if (shaderProgram == 0)
	{
		std::cout << "Error creating program object" << std::endl;
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);
	glLinkProgram(shaderProgram);
	glGetProgramiv(shaderProgram, GL_LINK_STATUS, &success);
	if (success == GL_FALSE)
	{
		std::cout << "Failed to link shader program!" << std::endl;
		GLint logLen;
		glGetProgramiv(shaderProgram, GL_INFO_LOG_LENGTH, &logLen);
		if (logLen > 0)
		{
			char *log = new char[logLen];
			GLsizei written;
			glGetProgramInfoLog(shaderProgram, logLen, &written, log);
			std::cout << "Shader program log:" << std::endl;
			std::cout << log << std::endl;
			delete[] log;
		}
	}

	// сначало бы отключить шейдеры!
	glDeleteShader(vertexShader);
	glDeleteShader(fragmentShader);

	return shaderProgram;
}

void InitPoints(vector<cv::Point3f> &points, GLuint &VAO)
{
	GLfloat *fPoints = new GLfloat[points.size()*3];
	GLfloat *fColorPoints = new GLfloat[points.size() * 3];
	int k = 0, ck = 0;
	for (auto it = points.begin(); it != points.end(); it++)
	{
		fPoints[k++] = it->x;
		fPoints[k++] = it->y;
		fPoints[k++] = it->z;

		fColorPoints[ck++] = 1.0f;
		fColorPoints[ck++] = 1.0f;
		fColorPoints[ck++] = 1.0f;
	}

	GLuint VBO[2];
	glGenBuffers(2, VBO);							// Сгенерируем 2 буфера

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);			// Сделаем 0 буфер текущим
	glBufferData(GL_ARRAY_BUFFER, points.size()*3*sizeof(GLfloat), fPoints, GL_STATIC_DRAW);  // положим туда данные о вершинах

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);			// Сделаем 1 буфер текущим
	glBufferData(GL_ARRAY_BUFFER, points.size() * 3 * sizeof(GLfloat), fColorPoints, GL_STATIC_DRAW); // положим туда информацию о цветах


	glGenVertexArrays(1, &VAO);
	glBindVertexArray(VAO);

	glEnableVertexAttribArray(0);		// Активируем буфер атрибутов шейдера
	glEnableVertexAttribArray(1);		// Активируем буфер атрибутов шейдера

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);


	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

	delete[] fPoints;
	delete[] fColorPoints;

}

void DrawPoints()
{
}

int CreateOpenGLWindow(vector<cv::Point3f> &points)
{
	InitGLFW();

	GLFWwindow *window = glfwCreateWindow(800, 600, "LearnOpenGL", nullptr, nullptr);
	if (window == nullptr)
	{
		std::cout << "Failed to create GLFW window" << std::endl;
		glfwTerminate();
		return -1;
	}
	glfwMakeContextCurrent(window);

	glewExperimental = GL_TRUE; // позволяет использовать новые технологии, могут возникнуть проблемы с core режимом, если не сделать ее true
	if (glewInit() != GLEW_OK)
	{
		std::cout << "Failed to initialize GLEW" << std::endl;
		return -1;
	}

	glfwSetKeyCallback(window, key_callback);

	int width, height;
	glfwGetFramebufferSize(window, &width, &height);
	glViewport(0, 0, width, height);

//---------------------------------------------------------
	GLuint shaderProgram = LoadShader(vertexShaderSource, fragmentShaderSource);
//---------------------------------------------------------
// Матрица проекции : 45° Угол обзора. 4:3 соотношение, дальность вида : 0.1 единиц <-> 100 единиц
	projection = glm::perspective(45.0f, 800.0f / 600.0f, 0.001f, 100.0f);
	// Матрица камеры
	glm::mat4 View = glm::lookAt(
		glm::vec3(points[points.size()-1].x, points[points.size() - 1].y, points[points.size() - 1].z), // Позиция в  (0,0,3)мировых координат
		//glm::vec3(0.0f, 0.0f, 3.0f), // Позиция в  (0,0,3)мировых координат
		glm::vec3(points[points.size() - 1].x, points[points.size() - 1].y, 0.0f), // И смотрит в центр экрана
		glm::vec3(0.0f, 1.0f, 0.0f)  // Верх камеры смотрит вверх
	);
	//// Матрица модели – единичная матрица. Модель находится в центре мировых координат
	glm::mat4 Model = glm::mat4(1.0f);  // Выставляем свое значение для каждой модели!
										// НАША МВП : Умножаем все наши три матрицы
	MVP = projection * View * Model;

	// для правильного отображения глубины
	glStencilOp(GL_KEEP, GL_KEEP, GL_REPLACE);
	glDepthFunc(GL_LEQUAL);
	glDepthRange(0.0, 1.0);
	glEnable(GL_DEPTH_TEST);
//----------------------------------------------------------
	GLuint axisVAO, axisVBO[2];
	InitAxis(axisVAO, axisVBO);
//----------------------------------------------------------
	GLfloat vertices[] = {
		-1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		1.0f, 1.0f, 0.0f,
		-1.0f, -1.0f, 0.0f,
		1.0f, -1.0f, 0.0f,
	};

	GLfloat colorVertices[] = {
		1.0f, 0.0f, 0.0f,
		0.0f, 1.0f, 0.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 0.0f, 1.0f,
		0.0f, 1.0f, 0.0f,
		1.0f, 0.0f, 0.0f,
	};

	GLuint VBO[2];
	glGenBuffers(2, VBO);							// Сгенерируем 2 буфера

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);			// Сделаем 0 буфер текущим
	glBufferData(GL_ARRAY_BUFFER, sizeof(vertices), vertices, GL_STATIC_DRAW);  // положим туда данные о вершинах

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);			// Сделаем 1 буфер текущим
	glBufferData(GL_ARRAY_BUFFER, sizeof(colorVertices), colorVertices, GL_STATIC_DRAW); // положим туда информацию о цветах


	GLuint VAO;
	glGenVertexArrays(1, &VAO);			
	glBindVertexArray(VAO);

	glEnableVertexAttribArray(0);		// Активируем буфер атрибутов шейдера
	glEnableVertexAttribArray(1);		// Активируем буфер атрибутов шейдера

	glBindBuffer(GL_ARRAY_BUFFER, VBO[0]);
	glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);

	glBindBuffer(GL_ARRAY_BUFFER, VBO[1]);
	glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, (GLvoid*)0);


	glBindBuffer(GL_ARRAY_BUFFER, 0);
	glBindVertexArray(0);

//----------------------------------------------------------
	GLuint pointsVAO;
	InitPoints(points, pointsVAO);

//----------------------------------------------------------

	GLuint matrixMVP = glGetUniformLocation(shaderProgram, "MVP");

//---------------------------------------------------------
	while (!glfwWindowShouldClose(window))
	{
		glfwPollEvents();

		glClearColor(0.2f, 0.3f, 0.3f, 1.0f);
		glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT);

		glUseProgram(shaderProgram);

		glUniformMatrix4fv(matrixMVP, 1, GL_FALSE, &MVP[0][0]);

		glBindVertexArray(VAO);
		glDrawArrays(GL_TRIANGLES, 0, 6);

		glBindVertexArray(axisVAO);	
		glLineWidth(2.0f);
		glDrawArrays(GL_LINES, 0, 6);

		glBindVertexArray(pointsVAO);
		glPointSize(3.0f);
		glDrawArrays(GL_POINTS, 0, points.size());

		glBindVertexArray(0);
		glfwSwapBuffers(window);
	}

//---------------------------------------------------------
	glDeleteVertexArrays(1, &VAO);
	glDeleteBuffers(2, VBO);

	glfwTerminate();
	return 0;
}