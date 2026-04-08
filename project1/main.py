from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

# Orbit -> 구면좌표계 사용 
# eye point 위치와 up vector 표현가능
g_cam_r = 5.0
g_cam_theta = np.radians(45)
g_cam_phi = np.radians(45)

# centor point 위치
g_cam_center = glm.vec3(0.0,0.0,0.0)
# 마우스
g_mouse_is_dragged = False
g_mouse_x_pos = 0.0
g_mouse_y_pos = 0.0
# 키보드 입력
g_z_is_pressed = False
g_x_is_pressed = False

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
    // 3D points in homogeneous coordinates
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);

    gl_Position = MVP * p3D_in_hcoord;

    vout_color = vec4(vin_color, 1.);
}
'''

g_fragment_shader_src = '''
#version 330 core

in vec4 vout_color;

out vec4 FragColor;

void main()
{
    FragColor = vout_color;
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

def button_callback(window, button, action, mod):
    global g_mouse_is_dragged, g_mouse_x_pos, g_mouse_y_pos
    if button==GLFW_MOUSE_BUTTON_LEFT:
        if action==GLFW_PRESS:
            g_mouse_is_dragged = True
            g_mouse_x_pos, g_mouse_y_pos = glfwGetCursorPos(window)
        elif action==GLFW_RELEASE:
            g_mouse_is_dragged = False

def key_callback(window, key, scancode, action, mods):
    global g_x_is_pressed, g_z_is_pressed
    if key == GLFW_KEY_X:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            g_x_is_pressed = True
        elif action == GLFW_RELEASE:
            g_x_is_pressed = False
            
    elif key == GLFW_KEY_Z:
        if action == GLFW_PRESS or action == GLFW_REPEAT:
            g_z_is_pressed = True
        elif action == GLFW_RELEASE:
            g_z_is_pressed = False

def cursor_callback(window, xpos, ypos):
    global g_cam_r, g_cam_theta, g_cam_phi, g_cam_center, g_mouse_is_dragged, g_mouse_x_pos, g_mouse_y_pos,g_x_is_pressed, g_z_is_pressed

    if g_mouse_is_dragged:
        # xpos는 parameter로 실시간 들어오는 현재 위치
        dx = xpos - g_mouse_x_pos
        dy = ypos - g_mouse_y_pos
        # glfwGetCursorPos(window)
        if g_x_is_pressed:
            # Pan action
            # xz 축만 이동, 현재 바라보고있는 방향에 대해서 -> theta 필요
            pan_sens = 0.01
            front_dir = glm.vec3(np.sin(g_cam_theta), 0.0, np.cos(g_cam_theta))
            right_dir = glm.vec3(np.sin(g_cam_theta - np.pi / 2), 0.0, np.cos(g_cam_theta- np.pi / 2))
            g_cam_center -= (front_dir * dy * pan_sens) - (right_dir * dx * pan_sens)
        elif g_z_is_pressed:
            # Zoom action
            # 현재 방향을 유지하고 거리만 바꾸기 -> r 필요
            # 마우스 양옆 움직임은 반영 X.
            zoom_sens = 0.01
            g_cam_r += dy * zoom_sens
            # r은 음수 값이 될 수 없음.
            g_cam_r = max(0.01, g_cam_r)
        else:
            # Orbit action
            orbit_sens = 0.01
            g_cam_theta -= dx * orbit_sens
            g_cam_phi += dy * orbit_sens
            # -90 ~ 90 으로만 제한. 예시에서 그렇게 구현되어있길래...
            g_cam_phi = max(-np.pi / 2 + 0.0001, min(g_cam_phi, np.pi / 2 - 0.0001))
    g_mouse_x_pos = xpos
    g_mouse_y_pos = ypos



def prepare_vao_oct():
    oct = []
    k = 1.0
    c = 0.1
    middles = [
        ([ k, k,  k], [ k, k, -k]),
        ([ k, k,  k], [-k, k,  k]),
        ([-k, k, -k], [-k, k,  k]),
        ([ k, k, -k], [-k, k, -k])
    ]
    top_bot = [[0.0, 0.0, 0.0],[0.0, 2 * k, 0.0]]
    for q in top_bot:
        for w,e in middles:
            oct.extend(q + [c,c,c])
            oct.extend(w + [c,c,c])
            oct.extend(e + [c,c,c])
            c += 0.1
    vertices = glm.array(glm.float32, *oct)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def draw_oct(vao,MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    # 
    glDrawArrays(GL_TRIANGLES, 0, 24)


def prepare_vao_frame():
    # prepare vertex data (in main memory)
    vertices = glm.array(glm.float32,
        # position        # color
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 3.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 3.0,  0.0, 0.0, 1.0, # z-axis end 
    )

    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)

    return VAO

def prepare_vao_grid():
    # consist of many "pos[2], col[2]"...
    # pos[2] = [x, y ,z] but it is xz grid so y should be zero.
    grid = []
    def rectangle(color,x,z):
        c = color
        #lower triangle 1-3
        grid.extend([x,0,z] + c)
        grid.extend([x+1,0,z] + c)
        grid.extend([x,0,z+1] + c)
        grid.extend([x+1,0,z] + c)
        grid.extend([x+1,0,z+1] + c)
        grid.extend([x,0,z+1] + c)
        # upper triangle 2-4
    wht = [0.8, 0.8, 0.8]
    blk = [0.2,0.2,0.2]
    for q in range(-5,5):
        for w in range(-5,5):
            if (abs(q+w) % 2) : rectangle(wht,q,w)
            else : rectangle(blk,q,w)

    vertices = glm.array(glm.float32, *grid)
    # create and activate VAO (vertex array object)
    VAO = glGenVertexArrays(1)  # create a vertex array object ID and store it to VAO variable
    glBindVertexArray(VAO)      # activate VAO

    # create and activate VBO (vertex buffer object)
    VBO = glGenBuffers(1)   # create a buffer object ID and store it to VBO variable
    glBindBuffer(GL_ARRAY_BUFFER, VBO)  # activate VBO as a vertex buffer object

    # copy vertex data to VBO
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW) # allocate GPU memory for and copy vertex data to the currently bound vertex buffer

    # configure vertex positions
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)

    # configure vertex colors
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    num = len(vertices) // 6
    return VAO, num

def draw_gird(vao, num, MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    # 
    glDrawArrays(GL_TRIANGLES, 0, num)
    #glDrawArrays(GL_LINES, 0, 6)


def main():
    # initialize glfw
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)   # OpenGL 3.3
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)  # Do not allow legacy OpenGl API calls
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE) # for macOS

    # create a window and OpenGL context
    window = glfwCreateWindow(1000, 1000, 'project1-2022056562', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # register event callbacks
    glfwSetMouseButtonCallback(window, button_callback)
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_callback)

    # load shaders
    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)

    # get uniform locations
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()
    vao_grid, num_grid_vertices = prepare_vao_grid()
    vao_oct = prepare_vao_oct()

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        # projection matrix        
        #P = glm.ortho(-1,1,-1,1,-1,1)
        P = glm.perspective(45,1,1,10)

        # view matrix
        # r, theta, phi -> 실제 위치값으로 변환
        eye_x = g_cam_center.x + g_cam_r * np.sin(g_cam_theta) * np.cos(g_cam_phi)
        eye_y = g_cam_center.y + g_cam_r * np.sin(g_cam_phi)
        eye_z = g_cam_center.z + g_cam_r * np.cos(g_cam_phi) * np.cos(g_cam_theta)
        # lookAt(eye,center,up)
        V = glm.lookAt(
            glm.vec3(eye_x, eye_y, eye_z),
            g_cam_center, 
            glm.vec3(0,1,0)
        )

        # current frame: P*V*I (now this is the world frame)
        I = glm.mat4()
        MVP = P*V*I
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw current frame --> xyz axis
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)
        
        # draw grid
        glBindVertexArray(vao_grid)
        draw_gird(vao_grid, num_grid_vertices, MVP,loc_MVP)
        # animating
        t = glfwGetTime()

        # rotation
        th = np.radians(t*90)
        R = glm.rotate(th, glm.vec3(0,1,0))

        # tranlation
        T = glm.translate(glm.vec3(np.sin(t), .2, 0.))

        # scaling
        S = glm.scale(glm.vec3(np.sin(t), np.sin(t), np.sin(t)))

        M = R
        # M = T
        # M = S
        # M = R @ T
        # M = T @ R

        # current frame: P*V*M
        MVP = P*V*M
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))

        # draw triangle w.r.t. the current frame
        draw_oct(vao_oct, MVP,loc_MVP)

        # draw current frame
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()