%% 1) 데이터 불러오기 및 전처리
clear; clc; close all

% HDF5 파일 및 경로 정의
filename   = 'sensor_20250613_sanguk_dance.h5';

path.time  = '/Sensor/Time/time';
path.imu1  = '/Sensor/IMU/imu1';   % R Thigh
path.imu2  = '/Sensor/IMU/imu2';   % R Shank
path.imu3  = '/Sensor/IMU/imu3';   % L Thigh
path.imu4  = '/Sensor/IMU/imu4';   % L Shank
path.imu5  = '/Sensor/IMU/imu5';   % Pelvic
path.imu6  = '/Sensor/IMU/imu6';   % Torso
path.imu7  = '/Sensor/IMU/imu7';   % R Upperarm
path.imu8  = '/Sensor/IMU/imu8';   % R Forearm
path.imu9  = '/Sensor/IMU/imu9';   % L Upperarm
path.imu10 = '/Sensor/IMU/imu10';  % L Forearm

path.button_a = '/Controller/button_a';
path.button_b = '/Controller/button_b';

% HDF5에서 원시 데이터 읽어오기
data.time  = h5read(filename, path.time);  
data.imu1  = h5read(filename, path.imu1);  
data.imu2  = h5read(filename, path.imu2);
data.imu3  = h5read(filename, path.imu3);
data.imu4  = h5read(filename, path.imu4);
data.imu5  = h5read(filename, path.imu5);
data.imu6  = h5read(filename, path.imu6);
data.imu7  = h5read(filename, path.imu7);
data.imu8  = h5read(filename, path.imu8);
data.imu9  = h5read(filename, path.imu9);
data.imu10 = h5read(filename, path.imu10);

data.button_a = h5read(filename, path.button_a);
data.button_b = h5read(filename, path.button_b);
data.label_a  = strcmp(data.button_a, 'TRUE');
data.label_b  = strcmp(data.button_b, 'TRUE');

rising_edge_a  = uint8([0; diff(data.label_a)  == 1]);
rising_edge_b  = uint8([0; diff(data.label_b)  == 1]);
rising_idx_a  = find(rising_edge_a  == true);
rising_idx_b  = find(rising_edge_b  == true);

% 시간 벡터(ms → s, 시작 0 기준)
time = double(data.time(rising_idx_a:rising_idx_b) - data.time(rising_idx_a)) / 1000;  
N    = numel(time);

% IMU 쿼터니언 정규화 (각 행 [w x y z])
q_rthigh_raw = quatnormalize(data.imu1(:,rising_idx_a:rising_idx_b)');  
q_rshank_raw = quatnormalize(data.imu2(:,rising_idx_a:rising_idx_b)');  
q_lthigh_raw = quatnormalize(data.imu3(:,rising_idx_a:rising_idx_b)');
q_lshank_raw = quatnormalize(data.imu4(:,rising_idx_a:rising_idx_b)');
q_pelvic_raw = quatnormalize(data.imu5(:,rising_idx_a:rising_idx_b)');
q_upbody_raw = quatnormalize(data.imu6(:,rising_idx_a:rising_idx_b)');
q_ruparm_raw = quatnormalize(data.imu7(:,rising_idx_a:rising_idx_b)');
q_rfoarm_raw = quatnormalize(data.imu8(:,rising_idx_a:rising_idx_b)');
q_luparm_raw = quatnormalize(data.imu9(:,rising_idx_a:rising_idx_b)');
q_lfoarm_raw = quatnormalize(data.imu10(:,rising_idx_a:rising_idx_b)');

clear data path rising_*

%% 2) 모델 Standing‐pose 보정용 쿼터니언 정의

showplot = false;

T_pose_time = 900;
[~, qD.rthigh] = plotQuatNearestAxes(q_rthigh_raw(T_pose_time,:), showplot);
[~, qD.rshank] = plotQuatNearestAxes(q_rshank_raw(T_pose_time,:), showplot);
[~, qD.lthigh] = plotQuatNearestAxes(q_lthigh_raw(T_pose_time,:), showplot);
[~, qD.lshank] = plotQuatNearestAxes(q_lshank_raw(T_pose_time,:), showplot);
[~, qD.pelvic] = plotQuatNearestAxes(q_pelvic_raw(T_pose_time,:), showplot);
[~, qD.upbody] = plotQuatNearestAxes(q_upbody_raw(T_pose_time,:), showplot);
[~, qD.ruparm] = plotQuatNearestAxes(q_ruparm_raw(T_pose_time,:), showplot);
[~, qD.rfoarm] = plotQuatNearestAxes(q_rfoarm_raw(T_pose_time,:), showplot);
[~, qD.luparm] = plotQuatNearestAxes(q_luparm_raw(T_pose_time,:), showplot);
[~, qD.lfoarm] = plotQuatNearestAxes(q_lfoarm_raw(T_pose_time,:), showplot);


%% 3) 첫 프레임(raw 1) 추출 및 고정 보정 상수 계산

qCorr.rthigh = quatmultiply(qD.rthigh, quatinv(q_rthigh_raw(T_pose_time,:)));
qCorr.rshank = quatmultiply(qD.rshank, quatinv(q_rshank_raw(T_pose_time,:)));
qCorr.lthigh = quatmultiply(qD.lthigh, quatinv(q_lthigh_raw(T_pose_time,:)));
qCorr.lshank = quatmultiply(qD.lshank, quatinv(q_lshank_raw(T_pose_time,:)));
qCorr.pelvic = quatmultiply(qD.pelvic, quatinv(q_pelvic_raw(T_pose_time,:)));
qCorr.upbody = quatmultiply(qD.upbody, quatinv(q_upbody_raw(T_pose_time,:)));
qCorr.ruparm = quatmultiply(qD.ruparm, quatinv(q_ruparm_raw(T_pose_time,:)));
qCorr.rfoarm = quatmultiply(qD.rfoarm, quatinv(q_rfoarm_raw(T_pose_time,:)));
qCorr.luparm = quatmultiply(qD.luparm, quatinv(q_luparm_raw(T_pose_time,:)));
qCorr.lfoarm = quatmultiply(qD.lfoarm, quatinv(q_lfoarm_raw(T_pose_time,:)));

%% 4) 모든 시점에 보정 곱하기 → 정렬된 쿼터니언

q_rthigh  = quatmultiply(qCorr.rthigh,  q_rthigh_raw);
q_rshank  = quatmultiply(qCorr.rshank,  q_rshank_raw);
q_lthigh  = quatmultiply(qCorr.lthigh,  q_lthigh_raw);
q_lshank  = quatmultiply(qCorr.lshank,  q_lshank_raw);
q_pelvic  = quatmultiply(qCorr.pelvic,  q_pelvic_raw);
q_upbody  = quatmultiply(qCorr.upbody,  q_upbody_raw);
q_ruparm  = quatmultiply(qCorr.ruparm,  q_ruparm_raw);
q_rfoarm  = quatmultiply(qCorr.rfoarm,  q_rfoarm_raw);
q_luparm  = quatmultiply(qCorr.luparm,  q_luparm_raw);
q_lfoarm  = quatmultiply(qCorr.lfoarm,  q_lfoarm_raw);


% (선택적) trunk 기반 yaw 필터링 → 지금은 'none'
highpass_or_remove = 'none';
if strcmp(highpass_or_remove, 'highpass')
    q_upbody   = quatyawhp(q_upbody);
    q_ruparm  = quatyawhp(q_ruparm);
    q_luparm  = quatyawhp(q_luparm);
    q_rfoarm  = quatyawhp(q_rfoarm);
    q_lfoarm  = quatyawhp(q_lfoarm);
elseif strcmp(highpass_or_remove, 'remove')
    q_upbody   = quatyawremove(q_upbody);
    q_ruparm  = quatyawremove(q_ruparm);
    q_luparm  = quatyawremove(q_luparm);
    q_rfoarm  = quatyawremove(q_rfoarm);
    q_lfoarm  = quatyawremove(q_lfoarm);
end

clear *_raw

%% 5) 절대(world) 회전행렬 계산

R_y180 = [-1 0 0 ; 0 1 0 ; 0 0 -1];

R_rthigh  = pagemtimes(R_y180, quat2rotm( quaternion(q_rthigh) ));
R_rshank  = pagemtimes(R_y180, quat2rotm( quaternion(q_rshank) ));
R_lthigh  = pagemtimes(R_y180, quat2rotm( quaternion(q_lthigh) ));
R_lshank  = pagemtimes(R_y180, quat2rotm( quaternion(q_lshank) ));
R_pelvic  = pagemtimes(R_y180, quat2rotm( quaternion(q_pelvic) ));
R_upbody  = pagemtimes(R_y180, quat2rotm( quaternion(q_upbody) ));
R_ruparm  = pagemtimes(R_y180, quat2rotm( quaternion(q_ruparm) ));
R_rfoarm  = pagemtimes(R_y180, quat2rotm( quaternion(q_rfoarm) ));
R_luparm  = pagemtimes(R_y180, quat2rotm( quaternion(q_luparm) ));
R_lfoarm  = pagemtimes(R_y180, quat2rotm( quaternion(q_lfoarm) ));

%% 6) 상대 쿼터니언 및 Euler 각도 산출 (관절 회전 확인용)

q_rshoulder = quatmultiply(quatconj(q_upbody), q_ruparm);
q_lshoulder = quatmultiply(quatconj(q_upbody), q_luparm);
q_relbow    = quatmultiply(quatconj(q_ruparm), q_rfoarm);
q_lelbow    = quatmultiply(quatconj(q_luparm), q_lfoarm);

euler_rshoulder  = rad2deg(quat2eul(q_rshoulder));
euler_lshoulder  = rad2deg(quat2eul(q_lshoulder));
euler_relbow = rad2deg(quat2eul(q_relbow));
euler_lelbow = rad2deg(quat2eul(q_lelbow));

figure();
subplot(411); plot(euler_rshoulder); title('Right Shoulder Euler (deg)');
subplot(412); plot(euler_lshoulder); title('Left  Shoulder Euler');
subplot(413); plot(euler_relbow);    title('Right Elbow Euler');
subplot(414); plot(euler_lelbow);    title('Left  Elbow Euler');

%% 7) 3-D Body-segment positions (FK, y-axis = link direction)
% ── 기본 파라미터 ───────────────────────────────────────────
hip_offset          = 0.14;   % 골반 중심 ↔ 좌/우 고관절 (x축)  [m]
shoulder_offset     = 0.24;   % 흉추 끝 ↔ 좌/우 어깨        [m]

link_length.trunk   = 0.35;   % T12 ~ C7(가상)                 [m]
link_length.thigh   = 0.45;   % 대퇴                          [m]
link_length.shank   = 0.50;   % 하퇴                          [m]
link_length.upperarm= 0.25;   % 상완                          [m]
link_length.forearm = 0.35;   % 전완                          [m]

v_origin  = [0; 0; 1];        % 골반 중심 (world origin)

% ── 세그먼트별 위치 배열 (3×N) 초기화 ───────────────────────
v_trunk   = zeros(3,N);   % T12 ~ C7 끝점
v_rshol   = zeros(3,N);   v_lshol = zeros(3,N);
v_ruparm  = zeros(3,N);   v_luparm = zeros(3,N);
v_rfoarm  = zeros(3,N);   v_lfoarm = zeros(3,N);

v_rhip    = zeros(3,N);   v_lhip   = zeros(3,N);
v_rthigh  = zeros(3,N);   v_lthigh = zeros(3,N);
v_rshank  = zeros(3,N);   v_lshank = zeros(3,N);

% ── FK 루프 ────────────────────────────────────────────────
for t = 1:N
    % ① trunk top (C7) : pelvis → y
    v_trunk(:,t) = v_origin + R_upbody(:,:,t) * [0; link_length.trunk; 0];

    % ② hip centres (좌/우) : pelvis 좌/우 x-offset
    v_rhip(:,t)  = v_origin + R_pelvic(:,:,t) * [ 0; 0; -hip_offset;];
    v_lhip(:,t)  = v_origin + R_pelvic(:,:,t) * [ 0; 0;  hip_offset;];

    % ③ shoulder centres : trunk top 좌/우 x-offset
    v_rshol(:,t) = v_trunk(:,t) + R_upbody(:,:,t) * [ shoulder_offset; 0; 0];
    v_lshol(:,t) = v_trunk(:,t) + R_upbody(:,:,t) * [-shoulder_offset; 0; 0];

    % ④ knee (thigh end) : hip → y
    v_rthigh(:,t)= v_rhip(:,t) + R_rthigh(:,:,t)*[0; -link_length.thigh; 0];
    v_lthigh(:,t)= v_lhip(:,t) + R_lthigh(:,:,t)*[0; -link_length.thigh; 0];

    % ⑤ ankle (shank end) : knee → y
    v_rshank(:,t)= v_rthigh(:,t)+ R_rshank(:,:,t)*[0; -link_length.shank; 0];
    v_lshank(:,t)= v_lthigh(:,t)+ R_lshank(:,:,t)*[0; -link_length.shank; 0];

    % ⑥ elbow (upper-arm end) : shoulder → y
    v_ruparm(:,t)= v_rshol(:,t)+ R_ruparm(:,:,t)*[0; -link_length.upperarm; 0];
    v_luparm(:,t)= v_lshol(:,t)+ R_luparm(:,:,t)*[0; -link_length.upperarm; 0];

    % ⑦ wrist (fore-arm end) : elbow → y
    v_rfoarm(:,t)= v_ruparm(:,t)+ R_rfoarm(:,:,t)*[0; -link_length.forearm; 0];
    v_lfoarm(:,t)= v_luparm(:,t)+ R_lfoarm(:,:,t)*[0; -link_length.forearm; 0];
end



%% 8) 3-D Animation
fig = figure('Position',[50 50 1000 900],'GraphicsSmoothing','on', ...
             'Name','IMU FK – Full-body');
axs = axes(fig,'Box','on');
view(190,10);
axis([-1 1 -1 1 -0.2 2]);  daspect([1 1 1]);
xlabel('X'); ylabel('Y'); zlabel('Z');

videorecord   = false;              % ← 필요 시 false 로
videoFileName = 'sangukdance_matlab.avi';
if videorecord
    vw = VideoWriter(videoFileName,'Motion JPEG AVI');
    vw.Quality   = 50;             % 1–100
    vw.FrameRate = 30;             % [fps]
    open(vw);
end

%─ Line objects (상체 + 하지) ───────────────────────────────
Line_trunk  = line('Color','k','LineWidth',5);
Line_pelvis = line('Color','k','LineWidth',5);
Line_ruparm = line('Color',[0.8 0 0],'LineWidth',5);
Line_luparm = line('Color',[0 0 0.8],'LineWidth',5);
Line_rfoarm = line('Color',[0.6 0.2 0],'LineWidth',5);
Line_lfoarm = line('Color',[0 0.2 0.6],'LineWidth',5);

Line_rthigh = line('Color',[0.8 0 0],'LineWidth',5); % dark-red
Line_lthigh = line('Color',[0 0 0.8],'LineWidth',5); % dark-green
Line_rshank = line('Color',[0.6 0.2 0],'LineWidth',5);
Line_lshank = line('Color',[0 0.2 0.6],'LineWidth',5);

%─ Axis handles ─────────────────────────────────────────────
axis_len = 0.1;
colors   = {'r','g','b'};
labels   = {'ground','pelvic','trunk','rthigh','lthigh','rshank','lshank', ...
            'ruparm','luparm','rfoarm','lfoarm'};
for i = 1:numel(labels)
    for j = 1:3
        AxisLines.(labels{i})(j) = line('Color',colors{j},'LineWidth',1.2);
    end
end

%─ Animation loop ───────────────────────────────────────────
for k = 1:3:N
    % ─ 링크 업데이트
    set(Line_trunk, 'XData',[v_origin(1) v_trunk(1,k)], ...
                    'YData',[v_origin(2) v_trunk(2,k)], ...
                    'ZData',[v_origin(3) v_trunk(3,k)]);

    set(Line_pelvis,'XData',[v_lhip(1,k) v_rhip(1,k)], ...
                    'YData',[v_lhip(2,k) v_rhip(2,k)], ...
                    'ZData',[v_lhip(3,k) v_rhip(3,k)]);

    set(Line_rthigh,'XData',[v_rhip(1,k) v_rthigh(1,k)], ...
                    'YData',[v_rhip(2,k) v_rthigh(2,k)], ...
                    'ZData',[v_rhip(3,k) v_rthigh(3,k)]);

    set(Line_lthigh,'XData',[v_lhip(1,k) v_lthigh(1,k)], ...
                    'YData',[v_lhip(2,k) v_lthigh(2,k)], ...
                    'ZData',[v_lhip(3,k) v_lthigh(3,k)]);

    set(Line_rshank,'XData',[v_rthigh(1,k) v_rshank(1,k)], ...
                    'YData',[v_rthigh(2,k) v_rshank(2,k)], ...
                    'ZData',[v_rthigh(3,k) v_rshank(3,k)]);

    set(Line_lshank,'XData',[v_lthigh(1,k) v_lshank(1,k)], ...
                    'YData',[v_lthigh(2,k) v_lshank(2,k)], ...
                    'ZData',[v_lthigh(3,k) v_lshank(3,k)]);

    set(Line_ruparm,'XData',[v_rshol(1,k) v_ruparm(1,k)], ...
                    'YData',[v_rshol(2,k) v_ruparm(2,k)], ...
                    'ZData',[v_rshol(3,k) v_ruparm(3,k)]);
    set(Line_luparm,'XData',[v_lshol(1,k) v_luparm(1,k)], ...
                    'YData',[v_lshol(2,k) v_luparm(2,k)], ...
                    'ZData',[v_lshol(3,k) v_luparm(3,k)]);

    set(Line_rfoarm,'XData',[v_ruparm(1,k) v_rfoarm(1,k)], ...
                    'YData',[v_ruparm(2,k) v_rfoarm(2,k)], ...
                    'ZData',[v_ruparm(3,k) v_rfoarm(3,k)]);
    set(Line_lfoarm,'XData',[v_luparm(1,k) v_lfoarm(1,k)], ...
                    'YData',[v_luparm(2,k) v_lfoarm(2,k)], ...
                    'ZData',[v_luparm(3,k) v_lfoarm(3,k)]);

    % ─ 축(로컬 XYZ) 업데이트
    base_pts = {[-0.9;-0.9;-0.1], v_origin, v_trunk(:,k), ...
                v_rhip(:,k), v_lhip(:,k), v_rthigh(:,k), v_lthigh(:,k), ...
                v_ruparm(:,k), v_luparm(:,k), v_rfoarm(:,k), v_lfoarm(:,k)};
    rot_mats = {eye(3), R_pelvic(:,:,k), R_upbody(:,:,k), ...
                R_pelvic(:,:,k), R_pelvic(:,:,k), ...
                R_rthigh(:,:,k), R_lthigh(:,:,k), ...
                R_ruparm(:,:,k), R_luparm(:,:,k), ...
                R_rfoarm(:,:,k), R_lfoarm(:,:,k)};

    for idx = 1:numel(labels)
        c = base_pts{idx};
        for ax = 1:3
            tip = c + axis_len * rot_mats{idx}(:,ax);
            set(AxisLines.(labels{idx})(ax),'XData',[c(1) tip(1)], ...
                                           'YData',[c(2) tip(2)], ...
                                           'ZData',[c(3) tip(3)]);
        end
    end
    drawnow;

    if videorecord
        writeVideo(vw, getframe(fig));
    end
end

if videorecord
    close(vw);
    fprintf('[INFO] Video saved → %s\n', videoFileName);
end

