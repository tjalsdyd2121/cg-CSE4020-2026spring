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

# fittable viewport 사용
g_P = glm.mat4()
g_height = 1000
g_width = 1000

# centor point 위치
g_cam_center = glm.vec3(0.0,0.0,0.0)
# 마우스
g_mouse_is_dragged = False
g_mouse_x_pos = 0.0
g_mouse_y_pos = 0.0
# 키보드 입력
g_z_is_pressed = False
g_x_is_pressed = False
g_f_is_pressed = False


# vertex shader 1 : grid, frame 처럼 절대적 색을 가짐.
g_vertex_shader_src_color_attribute = '''
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
# vertex shader 2 : 스스로 빛나는 물체. 태양!
g_vertex_shader_src_color_uniform = '''
#version 330 core
layout (location = 0) in vec3 vin_pos; 
out vec4 vout_color;
uniform mat4 MVP;
uniform vec3 color;
void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;
    vout_color = vec4(color, 1.);
}
'''
# vertex shader 1,2에 대한 공통 fragment shader.
g_fragment_shader_src_color = '''
#version 330 core
in vec4 vout_color;
out vec4 FragColor;
void main()
{
    FragColor = vout_color;
}
'''

# vertex shader 3 : lighting 되는 물체들.
# pos vector와 normal vector들을 obj 파일에서 받아와야함.
g_vertex_shader_src_light = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_normal; 

out vec3 vout_surface_pos;
out vec3 vout_normal;

uniform mat4 MVP;
uniform mat4 M;

void main()
{
    vec4 p3D_in_hcoord = vec4(vin_pos.xyz, 1.0);
    gl_Position = MVP * p3D_in_hcoord;

    vout_surface_pos = vec3(M * vec4(vin_pos, 1));
    vout_normal = normalize( mat3(inverse(transpose(M)) ) * vin_normal);
}
'''
# vertex shader 3에 대한 fragment shader.
g_fragment_shader_src_light = '''
#version 330 core

in vec3 vout_surface_pos;
in vec3 vout_normal;  // interpolated normal

out vec4 FragColor;

uniform vec3 view_pos;
// main에서 light_pos <= sun_pos 로 해주어야함.
uniform vec3 light_pos;
// 각 행성들에 대해서 color를 main 에서 지정해주기.
uniform vec3 material_color;


void main()
{
    // light and material properties
    // 태양은 일단 백색광
    vec3 light_color = vec3(1,1,1);
    float material_shininess = 32.0;

    // light components
    vec3 light_ambient = 0.1*light_color;
    vec3 light_diffuse = light_color;
    vec3 light_specular = light_color;

    // material components
    vec3 material_ambient = material_color;
    vec3 material_diffuse = material_color;
    vec3 material_specular = vec3(1,1,1);  // for non-metal material

    // ambient
    vec3 ambient = light_ambient * material_ambient;

    // for diffiuse and specular
    vec3 normal = normalize(vout_normal);
    vec3 surface_pos = vout_surface_pos;
    vec3 light_dir = normalize(light_pos - surface_pos);

    // diffuse
    float diff = max(dot(normal, light_dir), 0);
    vec3 diffuse = diff * light_diffuse * material_diffuse;

    // specular
    vec3 view_dir = normalize(view_pos - surface_pos);
    vec3 reflect_dir = reflect(-light_dir, normal);
    float spec = pow( max(dot(view_dir, reflect_dir), 0.0), material_shininess);
    vec3 specular = spec * light_specular * material_specular;

    vec3 color = ambient + diffuse + specular;
    FragColor = vec4(color, 1.);
}
'''

class Node:
    def __init__(self, parent, shape_transform, color, vao, vertex_count):
        self.parent = parent
        self.children = []
        if parent is not None:
            parent.children.append(self)

        self.transform = glm.mat4()
        self.global_transform = glm.mat4()

        self.shape_transform = shape_transform
        self.color = color
        self.vao = vao
        self.vertex_count = vertex_count

    def set_transform(self, transform):
        self.transform = transform

    def update_tree_global_transform(self):
        if self.parent is not None:
            self.global_transform = self.parent.get_global_transform() * self.transform
        else:
            self.global_transform = self.transform

        for child in self.children:
            child.update_tree_global_transform()

    def get_global_transform(self): return self.global_transform
    def get_shape_transform(self): return self.shape_transform
    def get_color(self): return self.color
    def get_vao(self): return self.vao
    def get_vertex_count(self): return self.vertex_count



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
            # right_dir은 + 해줘야지 생각처럼 작동함
            g_cam_center -= (front_dir * dy * pan_sens) - (right_dir * dx * pan_sens)
        elif g_z_is_pressed:
            # Zoom action
            # 현재 방향을 유지하고 거리만 바꾸기 -> r 필요
            # 마우스 양옆 움직임은 반영 X.
            zoom_sens = 0.01
            g_cam_r += dy * zoom_sens
            # r은 음수 값이 될 수 없음.
            # 너무 가까우면 좀 보기 그렇다.
            g_cam_r = max(0.1, g_cam_r)
        else:
            # Orbit action
            orbit_sens = 0.01
            g_cam_theta -= dx * orbit_sens
            g_cam_phi += dy * orbit_sens
            # -90 ~ 90 으로만 제한. 예시에서 그렇게 구현되어있고,
            # 그 이상으로 가버리면 상하 컨트롤이 정반대가 되어서 불편함
            g_cam_phi = max(-np.pi / 2 + 0.0001, min(g_cam_phi, np.pi / 2 - 0.0001))
    g_mouse_x_pos = xpos
    g_mouse_y_pos = ypos

def framebuffer_size_callback(window, width, height):
    global g_P, g_cam_r
    glViewport(0, 0, width, height)
    per_height = 10.
    per_width = per_height * width/height
    per_as = per_width/per_height
    g_P = glm.perspective(45, per_as, 0.05, 2* g_cam_r)

def prepare_vao_frame():
    # prepare vertex data (in main memory)
    # grid 그리는데 사용하기 위해 y를 맨뒤 순서로 바꾸기
    vertices = glm.array(glm.float32,
        # position        # color
         -3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis start
         3.0, 0.0, 0.0,  1.0, 0.0, 0.0, # x-axis end 
         0.0, 0.0, -3.0,  0.0, 0.0, 1.0, # z-axis start
         0.0, 0.0, 3.0,  0.0, 0.0, 1.0, # z-axis end
         0.0, -3.0, 0.0,  0.0, 1.0, 0.0, # y-axis start
         0.0, 3.0, 0.0,  0.0, 1.0, 0.0, # y-axis end 
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

def draw_frame(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_LINES, 0, 6)

def draw_grid(vao, MVP, loc_MVP):
    glBindVertexArray(vao)
    scale = 3
    MVP_ = MVP * glm.scale(glm.vec3(scale,scale,scale)) 
    for q in range(-15,14):
        for w in range(-14,15):
            MVP_grid = MVP_ * glm.translate(glm.vec3(1*q/scale, 0, 1*w/scale))
            glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP_grid))
            glDrawArrays(GL_LINES, 0, 4)

def main():
    global g_P, g_cam_r
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
    glfwSetFramebufferSizeCallback(window, framebuffer_size_callback)


    # load shaders
    shader_program = load_shaders(g_vertex_shader_src_color_attribute, g_fragment_shader_src_color)

    # get uniform locations
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    
    # prepare vaos
    vao_frame = prepare_vao_frame()

    # initialize projection matrix
    per_height = 10.
    # 1대1 비율로 생성
    per_width = per_height
    per_as = per_width/per_height
    g_P = glm.perspective(45, per_as, 0.05, 2 * g_cam_r)

    # loop until the user closes the window
    while not glfwWindowShouldClose(window):
        # render

        # enable depth test (we'll see details later)
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)

        glUseProgram(shader_program)

        width, height = glfwGetWindowSize(window)
        per_height = 10.
        per_width = per_height * width/height
        per_as = per_width/per_height
        g_P = glm.perspective(45, per_as, 0.05, 2* g_cam_r)

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
        MVP = g_P*V*I
        draw_frame(vao_frame, MVP, loc_MVP)
        draw_grid(vao_frame, MVP, loc_MVP)
        
        # draw check
        # animating
        t = glfwGetTime()

        # rotation
        th = np.radians(t*90)
        R = glm.rotate(th, glm.vec3(0,1,0))
        M = R


        # current frame: P*V*M
        MVP = g_P*V*M
        draw_frame(vao_frame, MVP, loc_MVP)
        # swap front and back buffers
        glfwSwapBuffers(window)

        # poll events
        glfwPollEvents()

    # terminate glfw
    glfwTerminate()

if __name__ == "__main__":
    main()