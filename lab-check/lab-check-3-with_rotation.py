from OpenGL.GL import *
from glfw.GLFW import *
import glm
import numpy as np

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec3 vout_color;

uniform mat2 M;

void main()
{
    vec2 rotated_xy = M * vin_pos.xy; // 세미콜론 필수 & 임시 변수 사용
    gl_Position = vec4(rotated_xy, vin_pos.z, 1.0); // 여기도 1 대신 1.0 권장
    vout_color = vin_color;
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec3 vout_color;
out vec4 FragColor;

uniform vec3 u_color;

void main()
{   
    FragColor = vec4(vout_color.z, vout_color.x, vout_color.y, 1) * vec4(u_color,1) + vec4(vout_color,1);

}
'''

def load_shaders(vertex_shader_source, fragment_shader_source):
    # build and compile our shader program
    # ------------------------------------
    
    # vertex shader 
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)    # create an empty shader object
    glShaderSource(vertex_shader, vertex_shader_source) # provide shader source code
    glCompileShader(vertex_shader)                      # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    # fragment shader
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)    # create an empty shader object
    glShaderSource(fragment_shader, fragment_shader_source) # provide shader source code
    glCompileShader(fragment_shader)                        # compile the shader object
    
    # check for shader compile errors
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    # link shaders
    shader_program = glCreateProgram()               # create an empty program object
    glAttachShader(shader_program, vertex_shader)    # attach the shader objects to the program object
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)                    # link the program object

    # check for linking errors
    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program    # return the shader program


def key_callback(window, key, scancode, action, mods):
    if key==GLFW_KEY_ESCAPE and action==GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE);

def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(800, 800, 'lab-check-3', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetKeyCallback(window, key_callback);

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    loc_m = glGetUniformLocation(shader_program, 'M') # find uniform's location
    #loc_u_pos = glGetUniformLocation(shader_program, 'u_pos') # find uniform's location
    # prepare vertex data (in main memory)
    # vertices = glm.array(glm.float32,
    #     -1.0, -1.0, 0.0, # left vertex x, y, z coordinates
    #      1.0, -1.0, 0.0, # right vertex x, y, z coordinates
    #      0.0,  1.0, 0.0  # top vertex x, y, z coordinates
    # )
    vertices = glm.array(glm.float32,
        # position        # color
        -1.0, -1.0, 0.0,  1.0, 0.0, 0.0, # left vertex
         1.0, -1.0, 0.0,  0.0, 1.0, 0.0, # right vertex
         0.0,  1.0, 0.0,  0.0, 0.0, 1.0, # top vertex
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex attributes
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)


    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render
        glClear(GL_COLOR_BUFFER_BIT)

        glUseProgram(shader_program)




        # update uniforms
        t = glfwGetTime()
        m = np.array([[np.cos(t), - np.sin(t)],
                      [np.sin(t), np.cos(t)]])
        glUniformMatrix2fv(loc_m,1,GL_TRUE,m)
        #fade = (glm.sin(t)+2) * .5        
        
        #print(glUniform3f(loc_u_pos, 1,1,1))
        #glUniformfv(loc_u_pos, [-1.0, -1.0, 0.0])




        glBindVertexArray(VAO)
        glDrawArrays(GL_TRIANGLES, 0, 3)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()
