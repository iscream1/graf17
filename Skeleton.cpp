//=============================================================================================
// Szamitogepes grafika hazi feladat keret. Ervenyes 2017-tol.
// A //~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// sorokon beluli reszben celszeru garazdalkodni, mert a tobbit ugyis toroljuk.
// A beadott program csak ebben a fajlban lehet, a fajl 1 byte-os ASCII karaktereket tartalmazhat.
// Tilos:
// - mast "beincludolni", illetve mas konyvtarat hasznalni
// - faljmuveleteket vegezni a printf-et kiv�ve
// - new operatort hivni a lefoglalt adat korrekt felszabaditasa nelkul
// - felesleges programsorokat a beadott programban hagyni
// - felesleges kommenteket a beadott programba irni a forrasmegjelolest kommentjeit kiveve
// ---------------------------------------------------------------------------------------------
// A feladatot ANSI C++ nyelvu forditoprogrammal ellenorizzuk, a Visual Studio-hoz kepesti elteresekrol
// es a leggyakoribb hibakrol (pl. ideiglenes objektumot nem lehet referencia tipusnak ertekul adni)
// a hazibeado portal ad egy osszefoglalot.
// ---------------------------------------------------------------------------------------------
// A feladatmegoldasokban csak olyan OpenGL/GLUT fuggvenyek hasznalhatok, amelyek az oran a feladatkiadasig elhangzottak
//
// NYILATKOZAT
// ---------------------------------------------------------------------------------------------
// Nev    :
// Neptun :
// ---------------------------------------------------------------------------------------------
// ezennel kijelentem, hogy a feladatot magam keszitettem, es ha barmilyen segitseget igenybe vettem vagy
// mas szellemi termeket felhasznaltam, akkor a forrast es az atvett reszt kommentekben egyertelmuen jeloltem.
// A forrasmegjeloles kotelme vonatkozik az eloadas foliakat es a targy oktatoi, illetve a
// grafhazi doktor tanacsait kiveve barmilyen csatornan (szoban, irasban, Interneten, stb.) erkezo minden egyeb
// informaciora (keplet, program, algoritmus, stb.). Kijelentem, hogy a forrasmegjelolessel atvett reszeket is ertem,
// azok helyessegere matematikai bizonyitast tudok adni. Tisztaban vagyok azzal, hogy az atvett reszek nem szamitanak
// a sajat kontribucioba, igy a feladat elfogadasarol a tobbi resz mennyisege es minosege alapjan szuletik dontes.
// Tudomasul veszem, hogy a forrasmegjeloles kotelmenek megsertese eseten a hazifeladatra adhato pontokat
// negativ elojellel szamoljak el es ezzel parhuzamosan eljaras is indul velem szemben.
//=============================================================================================

#define _USE_MATH_DEFINES
#include <stdio.h>
#include <stdlib.h>
#include <math.h>

#include <vector>

#if defined(__APPLE__)
#include <GLUT/GLUT.h>
#include <OpenGL/gl3.h>
#include <OpenGL/glu.h>
#else
#if defined(WIN32) || defined(_WIN32) || defined(__WIN32__)
#include <windows.h>
#endif
#include <GL/glew.h>		// must be downloaded
#include <GL/freeglut.h>	// must be downloaded unless you have an Apple
#endif

#include <iostream>
using std::cout;
using std::endl;

using std::vector;

const unsigned int windowWidth = 600, windowHeight = 600;

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// You are supposed to modify the code from here...

#ifndef M_PI
    #define M_PI 3.1415
#endif // M_PI

// OpenGL major and minor versions
int majorVersion = 3, minorVersion = 3;

void getErrorInfo(unsigned int handle) {
	int logLen;
	glGetShaderiv(handle, GL_INFO_LOG_LENGTH, &logLen);
	if (logLen > 0) {
		char * log = new char[logLen];
		int written;
		glGetShaderInfoLog(handle, logLen, &written, log);
		printf("Shader log:\n%s", log);
		delete log;
	}
}

// check if shader could be compiled
void checkShader(unsigned int shader, char * message) {
	int OK;
	glGetShaderiv(shader, GL_COMPILE_STATUS, &OK);
	if (!OK) {
		printf("%s!\n", message);
		getErrorInfo(shader);
	}
}

// check if shader could be linked
void checkLinking(unsigned int program) {
	int OK;
	glGetProgramiv(program, GL_LINK_STATUS, &OK);
	if (!OK) {
		printf("Failed to link shader program!\n");
		getErrorInfo(program);
	}
}

// vertex shader in GLSL
const char * vertexSource = R"(
	#version 330
    precision highp float;

	uniform mat4 MVP;			// Model-View-Projection matrix in row-major format

	layout(location = 0) in vec2 vertexPosition;	// Attrib Array 0
	layout(location = 1) in vec3 vertexColor;	    // Attrib Array 1
	out vec3 color;									// output attribute

	void main() {
		color = vertexColor;														// copy color from input to output
		gl_Position = vec4(vertexPosition.x, vertexPosition.y, 0, 1) * MVP; 		// transform to clipping space
	}
)";

// fragment shader in GLSL
const char * fragmentSource = R"(
	#version 330
    precision highp float;

	in vec3 color;				// variable input: interpolated color of vertex shader
	out vec4 fragmentColor;		// output that goes to the raster memory as told by glBindFragDataLocation

	void main() {
		fragmentColor = vec4(color, 1); // extend RGB to RGBA
	}
)";

// row-major matrix 4x4
struct mat4 {
	float m[4][4];
public:
	mat4() {}
	mat4(float m00, float m01, float m02, float m03,
		float m10, float m11, float m12, float m13,
		float m20, float m21, float m22, float m23,
		float m30, float m31, float m32, float m33) {
		m[0][0] = m00; m[0][1] = m01; m[0][2] = m02; m[0][3] = m03;
		m[1][0] = m10; m[1][1] = m11; m[1][2] = m12; m[1][3] = m13;
		m[2][0] = m20; m[2][1] = m21; m[2][2] = m22; m[2][3] = m23;
		m[3][0] = m30; m[3][1] = m31; m[3][2] = m32; m[3][3] = m33;
	}

	mat4 operator*(const mat4& right) {
		mat4 result;
		for (int i = 0; i < 4; i++) {
			for (int j = 0; j < 4; j++) {
				result.m[i][j] = 0;
				for (int k = 0; k < 4; k++) result.m[i][j] += m[i][k] * right.m[k][j];
			}
		}
		return result;
	}
	operator float*() { return &m[0][0]; }
};


// 3D point in homogeneous coordinates
struct vec4 {
	float v[4];

	vec4(float x = 0, float y = 0, float z = 0, float w = 1) {
		v[0] = x; v[1] = y; v[2] = z; v[3] = w;
	}

	vec4 operator*(const mat4& mat) {
		vec4 result;
		for (int j = 0; j < 4; j++) {
			result.v[j] = 0;
			for (int i = 0; i < 4; i++) result.v[j] += v[i] * mat.m[i][j];
		}
		return result;
	}

	vec4 operator-(const vec4 op)
	{
	    vec4 result;
	    for(int i=0;i<4;i++)
        {
            result.v[i]=v[i]-op.v[i];
        }
        return result;
	}

	vec4 operator+(const vec4 op)
	{
	    vec4 result;
	    for(int i=0;i<4;i++)
        {
            result.v[i]=v[i]+op.v[i];
        }
        return result;
	}

	vec4 operator/(const float a) {
		vec4 result;
		for (int i = 0; i < 4; i++) {
			result.v[i]=v[i]/a;
		}
		return result;
	}

	vec4 operator*(const float a) {
		vec4 result;
		for (int i = 0; i < 4; i++) {
			result.v[i]=v[i]*a;
		}
		return result;
	}
};

// 2D camera
struct Camera {
	float wCx, wCy;	// center in world coordinates
	float wWx, wWy;	// width and height in world coordinates
	int Vstate=false;
	float pa=0.0f, pb=0.0f;
public:
	Camera() {
		Animate(0);
	}

	void setVstate(int to)
	{
	    Vstate=to;
	}

	void setV(float a, float b)
	{
	    wCx=a;
        wCy=b;
	}

	int getVstate()
	{
	    return Vstate;
	}

	mat4 V() { // view matrix: translates the center to the origin
		return mat4(1,    0, 0, 0,
			        0,    1, 0, 0,
			        0,    0, 1, 0,
			     -wCx, -wCy, 0, 1);
	}

	mat4 P() { // projection matrix: scales it to be a square of edge length 2
		return mat4(2/wWx,    0, 0, 0,
			        0,    2/wWy, 0, 0,
			        0,        0, 1, 0,
			        0,        0, 0, 1);
	}

	mat4 Vinv() { // inverse view matrix
		return mat4(1,     0, 0, 0,
				    0,     1, 0, 0,
			        0,     0, 1, 0,
			        wCx, wCy, 0, 1);
	}

	mat4 Pinv() { // inverse projection matrix
		return mat4(wWx/2, 0,    0, 0,
			           0, wWy/2, 0, 0,
			           0,  0,    1, 0,
			           0,  0,    0, 1);
	}

	void Animate(float t) {
		wCx = 0; // 10 * cosf(t);
		wCy = 0;
		wWx = 20;
		wWy = 20;
	}
};

// 2D camera
Camera camera;

// handle of the shader program
unsigned int shaderProgram;

class Triangle {
	unsigned int vao;	// vertex array object id
	float sx, sy;		// scaling
	float wTx, wTy;		// translation
public:
	Triangle() {
		Animate(0);
	}

	void Create() {
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { -8, -8, -6, 10, 8, -2 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			         sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0
		glVertexAttribPointer(0,			// Attribute Array 0
			                  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { 1, 0, 0,  0, 1, 0,  0, 0, 1 };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Create(vec4 a, vec4 b, vec4 c, vec4 colorA, vec4 colorB, vec4 colorC){
		glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects

		// vertex coordinates: vbo[0] -> Attrib Array 0 -> vertexPosition of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		float vertexCoords[] = { a.v[0], a.v[1], b.v[0], b.v[1], c.v[0], c.v[1]};	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			         sizeof(vertexCoords), // number of the vbo in bytes
					 vertexCoords,		   // address of the data array on the CPU
					 GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0
		glVertexAttribPointer(0,			// Attribute Array 0
			                  2, GL_FLOAT,  // components/attribute, component type
							  GL_FALSE,		// not in fixed point format, do not normalized
							  0, NULL);     // stride and offset: it is tightly packed

		// vertex colors: vbo[1] -> Attrib Array 1 -> vertexColor of the vertex shader
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		float vertexColors[] = { colorA.v[0], colorA.v[1], colorA.v[2],  colorB.v[0], colorB.v[1], colorB.v[2],  colorC.v[0], colorC.v[1], colorC.v[2] };	// vertex data on the CPU
		glBufferData(GL_ARRAY_BUFFER, sizeof(vertexColors), vertexColors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
	}

	void Animate(float t) {
		sx = 1; // sinf(t);
		sy = 1; // cosf(t);
		wTx = 0; // 4 * cosf(t / 2);
		wTy = 0; // 4 * sinf(t / 2);
	}

	void Draw() {
		mat4 Mscale(sx, 0, 0, 0,
			        0, sy, 0, 0,
			        0, 0, 0, 0,
			        0, 0, 0, 1); // model matrix

		mat4 Mtranslate(1,   0,  0, 0,
			            0,   1,  0, 0,
			            0,   0,  0, 0,
			          wTx, wTy,  0, 1); // model matrix

		mat4 MVPTransform = Mscale * Mtranslate * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLES, 0, 3);	// draw a single triangle with vertices defined in vao
	}
};

class LineStrip {
	GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices
public:
	LineStrip() {
		nVertices = 0;
	}
	void Create() {
		glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
	}

	void AddPoint(float cX, float cY) {
		if (nVertices >= 20000) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices]     = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = 1; // red
		vertexData[5 * nVertices + 3] = 1; // green
		vertexData[5 * nVertices + 4] = 0; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}

	void Draw() {
		if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_LINE_STRIP, 0, nVertices);
		}
	}

	void clear()
	{
	    nVertices=0;
	}
};

/*class BezierSurface
{
protected:
    unsigned int vao;	// vertex array object id
    float colors[10000];
    float vertices[10000];
public:


	Star() {
	    Animate(0);
	}

	void Animate(float t) {
	}

	void Draw() {

		mat4 MVPTransform =R * M * camera.V() * camera.P();

		// set GPU uniform matrix variable MVP with the content of CPU variable MVPTransform
		int location = glGetUniformLocation(shaderProgram, "MVP");
		if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, MVPTransform); // set uniform variable MVP to the MVPTransform
		else printf("uniform MVP cannot be set\n");

		glBindVertexArray(vao);	// make the vao and its vbos active playing the role of the data source
		glDrawArrays(GL_TRIANGLE_FAN, 0, 16);	// draw a single triangle with vertices defined in vao
	}

    void Create(float r, float g, float b)
    {
        vertices[0]=0;
        vertices[1]=0;

        for(int i=0;i<400;i++)
        {
            vertices[i*2+2]=cos(fi);
            vertices[i*2+3]=sin(fi);
            fi+=deltaFi;
        }

        for(int i=0;i<7*N;i+=3)
        {
            colors[i]=r;
            colors[i+1]=g;
            colors[i+2]=b;
        }

        colors[N*3+3]=r;
        colors[N*3+4]=g;
        colors[N*3+5]=b;

        glGenVertexArrays(1, &vao);	// create 1 vertex array object
		glBindVertexArray(vao);		// make it active

		unsigned int vbo[2];		// vertex buffer objects
		glGenBuffers(2, &vbo[0]);	// Generate 2 vertex buffer objects
		glBindBuffer(GL_ARRAY_BUFFER, vbo[0]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER,      // copy to the GPU
			sizeof(vertices),  // number of the vbo in bytes
			vertices,		   // address of the data array on the CPU
			GL_STATIC_DRAW);	   // copy to that part of the memory which is not modified
		// Map Attribute Array 0 to the current bound vertex buffer (vbo[0])
		glEnableVertexAttribArray(0);
		// Data organization of Attribute Array 0
		glVertexAttribPointer(0,			// Attribute Array 0
			2, GL_FLOAT,  // components/attribute, component type
			GL_FALSE,		// not in fixed point format, do not normalized
			0, NULL);     // stride and offset: it is tightly packed
		glBindBuffer(GL_ARRAY_BUFFER, vbo[1]); // make it active, it is an array
		glBufferData(GL_ARRAY_BUFFER, sizeof(colors), colors, GL_STATIC_DRAW);	// copy to the GPU

		// Map Attribute Array 1 to the current bound vertex buffer (vbo[1])
		glEnableVertexAttribArray(1);  // Vertex position
		// Data organization of Attribute Array 1
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 0, NULL); // Attribute Array 1, components/attribute, component type, normalize?, tightly packed
    }

    void Move(vec4 to)
    {
        position=to;
    }

    vec4 getPos()
    {
        return position;
    }
};*/

class LagrangeCurve
{
    vector<vec4>  cps;	// control points
    vector<float> ts; 	// parameter (knot) values
    LineStrip ls;


    float L(unsigned int i, float t)
    {
        float Li = 1.0f;
        for(unsigned int j = 0; j < cps.size(); j++)
            if (j != i) Li *= (t - ts[j])/(ts[i]- ts[j]);
        return Li;
    }
public:
    LagrangeCurve() {
	}

	void Create() {
		ls.Create();
	}

    void AddPoint(float cX, float cY)
    {
        vec4 cp = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();

        ls.clear();
        float ti = cps.size();

        cps.push_back(cp);
        ts.push_back(ti);

        if(cps.size()>1)
        {
            for(unsigned int i=0;i<cps.size();i++)
            {
                float t1=ts[i];
                float t2=ts[i+1];
                for(float t=t1;t<=t2;t+=(t2-t1)/20)
                {
                    vec4 pos=r(t);
                    (pos=pos*camera.V() * camera.P());
                    ls.AddPoint(pos.v[0], pos.v[1]);
                }
            }
        }

    }

    vec4 r(float t)
    {
        vec4 rr(0, 0, 0);
        for(unsigned int i = 0; i < cps.size(); i++) rr = rr+ cps[i] * L(i,t);
        return rr;
    }


	void Draw()
	{
	    ls.Draw();
	}
};

class BezierCurve
{
    LineStrip ls;

    float B(int i, float t) {
        int n = cps.size()-1; // n deg polynomial = n+1 pts!
        float choose = 1;
      	for(int j = 1; j <= i; j++) choose *= (float)(n-j+1)/j;
      	return choose * pow(t, i) * pow(1-t, n-i);
   }

public:
    vector<vec4>  cps;	// control points
    vector<float> ts;

    BezierCurve() {
	}

	void Create(int x) {
	    ls.Create();
		for(int i=0;i<21;i++){
            AddControlPoint(x, -1.0f+((float)i/10));
		}
	}

    void AddControlPoint(float cX, float cY)
    {
        float cZ=((float)(rand()%1000))/1000;
        vec4 cp = vec4(cX, cY, cZ, 1) * camera.Pinv() * camera.Vinv();

        ls.clear();
        float ti = cps.size();

        cps.push_back(cp);
        ts.push_back(ti);

    }

    vec4 r(float t)
    {
        vec4 rr(0, 0);
        for(unsigned int i = 0; i < cps.size(); i++) rr = rr+ cps[i] * B(i,t);
        return rr;
    }

	void Draw()
	{
	    if(cps.size()>1)
        {
            for(unsigned int i=0;i<cps.size();i++)
            {
                float t1=ts[i];
                float t2=ts[i+1];
                for(float t=t1;t<=t2;t+=(t2-t1)/20)
                {
                    vec4 pos=r(t);
                    (pos=pos*camera.V() * camera.P());

                    ls.AddPoint(pos.v[0], pos.v[1]);
                }
            }
        }
	    ls.Draw();
	}
};

class BezierCurves
{
public:
    BezierCurve bcs[21];

    BezierCurves(){};

    void Create()
    {
        srand(10);
        for(int i=0;i<21;i++)
        {
            bcs[i].Create(-1.0f+((float)i/10));
        }
    }

    BezierCurve operator[](const int& idx)
    {
        return bcs[idx];
    }

    void Draw()
    {

        for(int i=0;i<21;i++)
        {
            bcs[i].Draw();
        }
    }
};

class Square{
public:
    Triangle t[2]; //triangles
};


class BezierSurface {
	/*GLuint vao, vbo;        // vertex array object, vertex buffer object
	float  vertexData[100000]; // interleaved data of coordinates and colors
	int    nVertices;       // number of vertices*/
	BezierCurves bezierCurves;
	Square squares[20][20];
public:
	BezierSurface() {
		//nVertices = 0;
	}
	void Create() {
	    bezierCurves.Create();
		/*glGenVertexArrays(1, &vao);
		glBindVertexArray(vao);

		glGenBuffers(1, &vbo); // Generate 1 vertex buffer object
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		// Enable the vertex attribute arrays
		glEnableVertexAttribArray(0);  // attribute array 0
		glEnableVertexAttribArray(1);  // attribute array 1
		// Map attribute array 0 to the vertex data of the interleaved vbo
		glVertexAttribPointer(0, 2, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(0)); // attribute array, components/attribute, component type, normalize?, stride, offset
		// Map attribute array 1 to the color data of the interleaved vbo
		glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 5 * sizeof(float), reinterpret_cast<void*>(2 * sizeof(float)));
        */
		addSquares();
	}

	/*void AddPoint(float cX, float cY, vec4 color) {
		if (nVertices >= 20000) return;

		vec4 wVertex = vec4(cX, cY, 0, 1) * camera.Pinv() * camera.Vinv();
		// fill interleaved data
		vertexData[5 * nVertices]     = wVertex.v[0];
		vertexData[5 * nVertices + 1] = wVertex.v[1];
		vertexData[5 * nVertices + 2] = color.v[0]; // red
		vertexData[5 * nVertices + 3] = color.v[1]; // green
		vertexData[5 * nVertices + 4] = color.v[2]; // blue
		nVertices++;
		// copy data to the GPU
		glBindBuffer(GL_ARRAY_BUFFER, vbo);
		glBufferData(GL_ARRAY_BUFFER, nVertices * 5 * sizeof(float), vertexData, GL_DYNAMIC_DRAW);
	}*/

	void addSquares()
	{
	    for(int i=0;i<20;i++)
            for(int j=0;j<20;j++)
            {
                float za=bezierCurves[i].   r(j).   v[2]; //bal lenn
                float zb=bezierCurves[i+1]. r(j).   v[2]; //jobb lenn
                float zc=bezierCurves[i+1]. r(j+1). v[2]; //jobb fenn
                float zd=bezierCurves[i].   r(j+1). v[2]; //bal fenn
                //cout<<za<<" "<<zb<<" "<<zc<<" "<<zd;
                /*AddPoint(-1.0f+((float)i/10),       -1.0f+((float)j/10),        vec4(float(144/255), za, 1));
                AddPoint(-1.0f+((float)(i+1)/10),   -1.0f+((float)j/10),        vec4(float(144/255), zb, 1));
                AddPoint(-1.0f+((float)(i+1)/10),   -1.0f+((float)(j+1)/10),    vec4(float(144/255), zc, 1));
                AddPoint(-1.0f+((float)i/10),       -1.0f+((float)(j+1)/10),    vec4(float(144/255), zd, 1));*/
                vec4 a=vec4(-10+i, -10+j, 0);
                vec4 b=vec4(-10+i+1, -10+j, 0);
                vec4 c=vec4(-10+i+1, -10+j+1, 0);
                vec4 d=vec4(-10+i, -10+j+1, 0);
                //cout<<a.v[0]<<" "<<a.v[1]<<endl<<b.v[0]<<" "<<b.v[1]<<endl<<c.v[0]<<" "<<c.v[1]<<endl<<d.v[0]<<" "<<d.v[1]<<endl<<endl;

                vec4 colorA=vec4(float(144/255), za/10, 1);
                vec4 colorB=vec4(float(144/255), zb/10, 1);
                vec4 colorC=vec4(float(144/255), zc/10, 1);
                vec4 colorD=vec4(float(144/255), zd/10, 1);
                squares[i][j].t[0].Create(a, b, c, colorA, colorB, colorC);
                squares[i][j].t[1].Create(a, c, d, colorA, colorC, colorD);
            }
	}

	void Draw() {
	    //bezierCurves.Draw();
		/*if (nVertices > 0) {
			mat4 VPTransform = camera.V() * camera.P();

			int location = glGetUniformLocation(shaderProgram, "MVP");
			if (location >= 0) glUniformMatrix4fv(location, 1, GL_TRUE, VPTransform);
			else printf("uniform MVP cannot be set\n");

			glBindVertexArray(vao);
			glDrawArrays(GL_TRIANGLE_STRIP, 0, nVertices);
		}*/
		for(int i=0;i<20;i++)
            for(int j=0;j<20;j++)
            {
                squares[i][j].t[0].Draw();
                squares[i][j].t[1].Draw();
            }
	}
};

// The virtual world: collection of two objects
Triangle triangle;
LagrangeCurve lineStrip;
BezierSurface bezierSurface;

// Initialization, create an OpenGL context
void onInitialization() {
	glViewport(0, 0, windowWidth, windowHeight);

	// Create objects by setting up their vertex data on the GPU
	lineStrip.Create();
	triangle.Create();
	bezierSurface.Create();


	// Create vertex shader from string
	unsigned int vertexShader = glCreateShader(GL_VERTEX_SHADER);
	if (!vertexShader) {
		printf("Error in vertex shader creation\n");
		exit(1);
	}
	glShaderSource(vertexShader, 1, &vertexSource, NULL);
	glCompileShader(vertexShader);
	checkShader(vertexShader, "Vertex shader error");

	// Create fragment shader from string
	unsigned int fragmentShader = glCreateShader(GL_FRAGMENT_SHADER);
	if (!fragmentShader) {
		printf("Error in fragment shader creation\n");
		exit(1);
	}
	glShaderSource(fragmentShader, 1, &fragmentSource, NULL);
	glCompileShader(fragmentShader);
	checkShader(fragmentShader, "Fragment shader error");

	// Attach shaders to a single program
	shaderProgram = glCreateProgram();
	if (!shaderProgram) {
		printf("Error in shader program creation\n");
		exit(1);
	}
	glAttachShader(shaderProgram, vertexShader);
	glAttachShader(shaderProgram, fragmentShader);

	// Connect the fragmentColor to the frame buffer memory
	glBindFragDataLocation(shaderProgram, 0, "fragmentColor");	// fragmentColor goes to the frame buffer memory

	// program packaging
	glLinkProgram(shaderProgram);
	checkLinking(shaderProgram);
	// make this program run
	glUseProgram(shaderProgram);
}

void onExit() {
	glDeleteProgram(shaderProgram);
	printf("exit");
}

// Window has become invalid: Redraw
void onDisplay() {
	glClearColor(0, 0, 0, 0);							// background color
	glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT); // clear the screen

	triangle.Draw();
	bezierSurface.Draw();
	lineStrip.Draw();


	glutSwapBuffers();									// exchange the two buffers
}

// Key of ASCII code pressed
void onKeyboard(unsigned char key, int pX, int pY) {
	if (key == 'd') glutPostRedisplay();         // if d, invalidate display, i.e. redraw
}

// Key of ASCII code released
void onKeyboardUp(unsigned char key, int pX, int pY) {

}

// Mouse click event
void onMouse(int button, int state, int pX, int pY) {
	if (button == GLUT_LEFT_BUTTON && state == GLUT_DOWN) {  // GLUT_LEFT_BUTTON / GLUT_RIGHT_BUTTON and GLUT_DOWN / GLUT_UP
		float cX = 2.0f * pX / windowWidth - 1;	// flip y axis
		float cY = 1.0f - 2.0f * pY / windowHeight;
		lineStrip.AddPoint(cX, cY);
		glutPostRedisplay();     // redraw
	}
}

// Move mouse with key pressed
void onMouseMotion(int pX, int pY) {
}

// Idle event indicating that some time elapsed: do animation here
void onIdle() {
	long time = glutGet(GLUT_ELAPSED_TIME); // elapsed time since the start of the program
	float sec = time / 1000.0f;				// convert msec to sec
	camera.Animate(sec);					// animate the camera
	triangle.Animate(sec);					// animate the triangle object
	glutPostRedisplay();					// redraw the scene
}

//~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~~
// Do not touch the code below this line

int main(int argc, char * argv[]) {
	glutInit(&argc, argv);
#if !defined(__APPLE__)
	glutInitContextVersion(majorVersion, minorVersion);
#endif
	glutInitWindowSize(windowWidth, windowHeight);				// Application window is initially of resolution 600x600
	glutInitWindowPosition(100, 100);							// Relative location of the application window
#if defined(__APPLE__)
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH | GLUT_3_3_CORE_PROFILE);  // 8 bit R,G,B,A + double buffer + depth buffer
#else
	glutInitDisplayMode(GLUT_RGBA | GLUT_DOUBLE | GLUT_DEPTH);
#endif
	glutCreateWindow(argv[0]);

#if !defined(__APPLE__)
	glewExperimental = true;	// magic
	glewInit();
#endif

	printf("GL Vendor    : %s\n", glGetString(GL_VENDOR));
	printf("GL Renderer  : %s\n", glGetString(GL_RENDERER));
	printf("GL Version (string)  : %s\n", glGetString(GL_VERSION));
	glGetIntegerv(GL_MAJOR_VERSION, &majorVersion);
	glGetIntegerv(GL_MINOR_VERSION, &minorVersion);
	printf("GL Version (integer) : %d.%d\n", majorVersion, minorVersion);
	printf("GLSL Version : %s\n", glGetString(GL_SHADING_LANGUAGE_VERSION));

	onInitialization();

	glutDisplayFunc(onDisplay);                // Register event handlers
	glutMouseFunc(onMouse);
	glutIdleFunc(onIdle);
	glutKeyboardFunc(onKeyboard);
	glutKeyboardUpFunc(onKeyboardUp);
	glutMotionFunc(onMouseMotion);

	glutMainLoop();
	onExit();
	return 1;
}

