from OpenGL.GL import *
from glfw.GLFW import *
import glm
import ctypes
import numpy as np

g_cam_azimuth = np.radians(45)   # 방위각 (좌우 회전)
g_cam_elevation = np.radians(35) # 앙각 (상하 회전)
g_cam_distance = 5.0             # 카메라와 타겟 사이의 거리 (Zoom)
g_cam_target = glm.vec3(0.0, 0.0, 0.0) # 카메라가 바라보는 중심점 (Pan)

# 마우스 상태 변수
g_is_dragging = False
g_last_x = 0.0
g_last_y = 0.0

g_vertex_shader_src = '''
#version 330 core

layout (location = 0) in vec3 vin_pos; 
layout (location = 1) in vec3 vin_color; 

out vec4 vout_color;

uniform mat4 MVP;

void main()
{
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
    vertex_shader = glCreateShader(GL_VERTEX_SHADER)
    glShaderSource(vertex_shader, vertex_shader_source)
    glCompileShader(vertex_shader)
    
    success = glGetShaderiv(vertex_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(vertex_shader)
        print("ERROR::SHADER::VERTEX::COMPILATION_FAILED\n" + infoLog.decode())
        
    fragment_shader = glCreateShader(GL_FRAGMENT_SHADER)
    glShaderSource(fragment_shader, fragment_shader_source)
    glCompileShader(fragment_shader)
    
    success = glGetShaderiv(fragment_shader, GL_COMPILE_STATUS)
    if (not success):
        infoLog = glGetShaderInfoLog(fragment_shader)
        print("ERROR::SHADER::FRAGMENT::COMPILATION_FAILED\n" + infoLog.decode())

    shader_program = glCreateProgram()
    glAttachShader(shader_program, vertex_shader)
    glAttachShader(shader_program, fragment_shader)
    glLinkProgram(shader_program)

    success = glGetProgramiv(shader_program, GL_LINK_STATUS)
    if (not success):
        infoLog = glGetProgramInfoLog(shader_program)
        print("ERROR::SHADER::PROGRAM::LINKING_FAILED\n" + infoLog.decode())
        
    glDeleteShader(vertex_shader)
    glDeleteShader(fragment_shader)

    return shader_program

# --- 콜백 함수들 ---

def key_callback(window, key, scancode, action, mods):
    # 프로그램 종료 용도
    if key == GLFW_KEY_ESCAPE and action == GLFW_PRESS:
        glfwSetWindowShouldClose(window, GLFW_TRUE)

def scroll_callback(window, xoffset, yoffset):
    # 일반 마우스 휠 사용자나 두 손가락 스크롤을 위한 기본 줌 기능도 남겨둡니다
    global g_cam_distance
    zoom_sensitivity = 0.5
    g_cam_distance -= yoffset * zoom_sensitivity
    g_cam_distance = max(0.1, g_cam_distance) 

def cursor_pos_callback(window, xpos, ypos):
    global g_cam_azimuth, g_cam_elevation, g_cam_target, g_cam_distance, g_last_x, g_last_y, g_is_dragging
    
    if g_is_dragging:
        dx = xpos - g_last_x
        dy = ypos - g_last_y
        
        # Z키, X키 눌림 상태 확인
        is_z_pressed = (glfwGetKey(window, GLFW_KEY_Z) == GLFW_PRESS)
        is_x_pressed = (glfwGetKey(window, GLFW_KEY_X) == GLFW_PRESS)
                           
        if is_z_pressed:
            # --- Zoom (Z + 상하 드래그) ---
            zoom_drag_sensitivity = 0.02
            # 마우스를 위로 올리면(음수 dy) 줌인, 아래로 내리면(양수 dy) 줌아웃
            g_cam_distance += dy * zoom_drag_sensitivity
            g_cam_distance = max(0.1, g_cam_distance) # 최소 거리 제한
            
        elif is_x_pressed:
            # --- Pan (X + 드래그, 수평 XZ 평면 이동) ---
            pan_sensitivity = 0.002 * g_cam_distance 
            
            # Y축(높이) 성분이 0인 순수 수평 Forward 및 Right 벡터 계산
            forward = glm.vec3(np.sin(g_cam_azimuth), 0.0, np.cos(g_cam_azimuth))
            right = glm.normalize(glm.cross(forward, glm.vec3(0, 1, 0)))
            
            # 좌우 드래그(dx) -> Right 벡터 방향 이동
            # 상하 드래그(dy) -> Forward 벡터 방향으로 앞뒤 이동 
            g_cam_target -= right * dx * pan_sensitivity
            g_cam_target -= forward * dy * pan_sensitivity 
            
        else:
            # --- Orbit (일반 드래그) ---
            orbit_sensitivity = 0.01
            g_cam_azimuth -= dx * orbit_sensitivity
            g_cam_elevation += dy * orbit_sensitivity
            
            # Gimbal lock 방지 및 위아래로 완전히 뒤집히지 않도록 제한
            g_cam_elevation = np.clip(g_cam_elevation, -np.pi/2 + 0.01, np.pi/2 - 0.01)
            
    g_last_x = xpos
    g_last_y = ypos

def mouse_button_callback(window, button, action, mods):
    global g_is_dragging, g_last_x, g_last_y
    if button == GLFW_MOUSE_BUTTON_LEFT:
        if action == GLFW_PRESS:
            g_is_dragging = True
            g_last_x, g_last_y = glfwGetCursorPos(window)
        elif action == GLFW_RELEASE:
            g_is_dragging = False

# --- VAO 준비 함수들 ---

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
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def draw_oct(vao,MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, 24)

def prepare_vao_frame():
    vertices = glm.array(glm.float32,
         0.0, 0.0, 0.0,  1.0, 0.0, 0.0, 
         1.0, 0.0, 0.0,  1.0, 0.0, 0.0,  
         0.0, 0.0, 0.0,  0.0, 1.0, 0.0, 
         0.0, 1.0, 0.0,  0.0, 1.0, 0.0, 
         0.0, 0.0, 0.0,  0.0, 0.0, 1.0, 
         0.0, 0.0, 1.0,  0.0, 0.0, 1.0,  
    )
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    return VAO

def prepare_vao_grid():
    grid = []
    def rectangle(color,x,z):
        c = color
        grid.extend([x,0,z] + c)
        grid.extend([x+1,0,z] + c)
        grid.extend([x,0,z+1] + c)
        grid.extend([x+1,0,z] + c)
        grid.extend([x+1,0,z+1] + c)
        grid.extend([x,0,z+1] + c)
        
    wht = [0.8, 0.8, 0.8]
    blk = [0.2, 0.2, 0.2]
    for q in range(-5,5):
        for w in range(-5,5):
            if (abs(q+w) % 2) : rectangle(wht,q,w)
            else : rectangle(blk,q,w)

    vertices = glm.array(glm.float32, *grid)
    VAO = glGenVertexArrays(1)
    glBindVertexArray(VAO)
    VBO = glGenBuffers(1)
    glBindBuffer(GL_ARRAY_BUFFER, VBO)
    glBufferData(GL_ARRAY_BUFFER, vertices.nbytes, vertices.ptr, GL_STATIC_DRAW)
    glVertexAttribPointer(0, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), None)
    glEnableVertexAttribArray(0)
    glVertexAttribPointer(1, 3, GL_FLOAT, GL_FALSE, 6 * glm.sizeof(glm.float32), ctypes.c_void_p(3*glm.sizeof(glm.float32)))
    glEnableVertexAttribArray(1)
    num = len(vertices) // 6
    return VAO, num

def draw_gird(vao, num, MVP, loc_MVP):
    glBindVertexArray(vao)
    glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))
    glDrawArrays(GL_TRIANGLES, 0, num)

def main():
    if not glfwInit():
        return
    glfwWindowHint(GLFW_CONTEXT_VERSION_MAJOR, 3)
    glfwWindowHint(GLFW_CONTEXT_VERSION_MINOR, 3)
    glfwWindowHint(GLFW_OPENGL_PROFILE, GLFW_OPENGL_CORE_PROFILE)
    glfwWindowHint(GLFW_OPENGL_FORWARD_COMPAT, GL_TRUE)

    window = glfwCreateWindow(1000, 1000, 'project1-OrbitCamera', None, None)
    if not window:
        glfwTerminate()
        return
    glfwMakeContextCurrent(window)

    # --- 콜백 등록 ---
    glfwSetKeyCallback(window, key_callback)
    glfwSetCursorPosCallback(window, cursor_pos_callback)
    glfwSetMouseButtonCallback(window, mouse_button_callback)
    glfwSetScrollCallback(window, scroll_callback)

    shader_program = load_shaders(g_vertex_shader_src, g_fragment_shader_src)
    loc_MVP = glGetUniformLocation(shader_program, 'MVP')
    
    vao_frame = prepare_vao_frame()
    vao_grid, num_grid_vertices = prepare_vao_grid()
    vao_oct = prepare_vao_oct()

    while not glfwWindowShouldClose(window):
        glClear(GL_COLOR_BUFFER_BIT | GL_DEPTH_BUFFER_BIT)
        glEnable(GL_DEPTH_TEST)
        glUseProgram(shader_program)

        P = glm.perspective(45, 1, 1, 100)

        # --- View Matrix 계산 (구면 좌표계 -> 데카르트 좌표계 변환) ---
        cam_x = g_cam_target.x + g_cam_distance * np.cos(g_cam_elevation) * np.sin(g_cam_azimuth)
        cam_y = g_cam_target.y + g_cam_distance * np.sin(g_cam_elevation)
        cam_z = g_cam_target.z + g_cam_distance * np.cos(g_cam_elevation) * np.cos(g_cam_azimuth)
        
        V = glm.lookAt(
            glm.vec3(cam_x, cam_y, cam_z), # 카메라 위치
            g_cam_target,                  # 바라보는 타겟 위치
            glm.vec3(0, 1, 0)              # Up 벡터
        )

        I = glm.mat4()
        MVP = P * V * I
        glUniformMatrix4fv(loc_MVP, 1, GL_FALSE, glm.value_ptr(MVP))

        # 프레임 및 그리드 그리기
        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)
        
        glBindVertexArray(vao_grid)
        draw_gird(vao_grid, num_grid_vertices, MVP, loc_MVP)
        
        # 오브젝트 애니메이션 및 렌더링
        t = glfwGetTime()
        th = np.radians(t * 90)
        R = glm.rotate(th, glm.vec3(0, 1, 0))
        M = R
        MVP = P * V * M
        
        draw_oct(vao_oct, MVP, loc_MVP)

        glBindVertexArray(vao_frame)
        glDrawArrays(GL_LINES, 0, 6)

        glfwSwapBuffers(window)
        glfwPollEvents()

    glfwTerminate()

if __name__ == "__main__":
    main()