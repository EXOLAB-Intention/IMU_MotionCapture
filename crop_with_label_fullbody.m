function crop_with_label_upperbody(h5Name, manual_check)

% [EMG]
% EMG L1: Left Trapezius (승모근) 
% EMG L2: Left Anterior Deltoid (전면삼각근)
% EMG L3: Left Latissimus dorsi (광배근)
% EMG L4: Left Elector spinae (기립근)
% 
% EMG R1: Right Trapezius
% EMG R2: Right Anterior Deltoid
% EMG R3: Right Latissimus dorsi
% EMG R4: Right Elector spinae
% 


    
    %% Load the h5 file
    filename = h5Name;
    
    time = '/Sensor/Time/time';
    emgL1 = '/Sensor/EMG/emgL1';
    emgL2 = '/Sensor/EMG/emgL2';
    emgL3 = '/Sensor/EMG/emgL3';
    emgL4 = '/Sensor/EMG/emgL4';
    emgR1 = '/Sensor/EMG/emgR1';
    emgR2 = '/Sensor/EMG/emgR2';
    emgR3 = '/Sensor/EMG/emgR3';
    emgR4 = '/Sensor/EMG/emgR4';

    imu1  = '/Sensor/IMU/imu1';   % R Thigh
    imu2  = '/Sensor/IMU/imu2';   % R Shank
    imu3  = '/Sensor/IMU/imu3';   % L Thigh
    imu4  = '/Sensor/IMU/imu4';   % L Shank
    imu5  = '/Sensor/IMU/imu5';   % Pelvic
    imu6  = '/Sensor/IMU/imu6';   % Torso
    imu7  = '/Sensor/IMU/imu7';   % R Upperarm
    imu8  = '/Sensor/IMU/imu8';   % R Forearm
    imu9  = '/Sensor/IMU/imu9';   % L Upperarm
    imu10 = '/Sensor/IMU/imu10';  % L Forearm
    
    button_ok = '/Controller/button_ok';
    button_a = '/Controller/button_a';
    button_b = '/Controller/button_b';
    
    data_time = h5read(filename, time);
    data_emgL1 = (h5read(filename, emgL1));
    data_emgL2 = (h5read(filename, emgL2));
    data_emgL3 = (h5read(filename, emgL3));
    data_emgL4 = (h5read(filename, emgL4));
    data_emgR1 = (h5read(filename, emgR1));
    data_emgR2 = (h5read(filename, emgR2));
    data_emgR3 = (h5read(filename, emgR3));
    data_emgR4 = (h5read(filename, emgR4));
    
    data_imu1  = (h5read(filename, imu1))';
    data_imu2  = (h5read(filename, imu2))';
    data_imu3  = (h5read(filename, imu3))';
    data_imu4  = (h5read(filename, imu4))';
    data_imu5  = (h5read(filename, imu5))';
    data_imu6  = (h5read(filename, imu6))';
    data_imu7  = (h5read(filename, imu7))';
    data_imu8  = (h5read(filename, imu8))';
    data_imu9  = (h5read(filename, imu9))';
    data_imu10 = (h5read(filename, imu10))';
        
    data_button_ok = (h5read(filename, button_ok));
    data_button_a  = (h5read(filename, button_a));
    data_button_b  = (h5read(filename, button_b));
    
    data_label_ok = strcmp(data_button_ok, 'TRUE');
    data_label_a  = strcmp(data_button_a, 'TRUE');
    data_label_b  = strcmp(data_button_b, 'TRUE');    
    
    %%
    
    rising_edge_ok = uint8([0; diff(data_label_ok) == 1]);
    falling_edge_ok = uint8([0; diff(data_label_ok) == -1]);
    rising_edge_a  = uint8([0; diff(data_label_a)  == 1]);
    rising_edge_b  = uint8([0; diff(data_label_b)  == 1]);
       
    rising_idx_ok = find(rising_edge_ok == true);
    rising_idx_a  = find(rising_edge_a  == true);
    rising_idx_b  = find(rising_edge_b  == true);

    valid_pairs = [];
    
    for i = 1:length(rising_idx_a)
        start_idx = rising_idx_a(i);
        next_stop_idx = rising_idx_b(find(rising_idx_b > start_idx, 1, 'first'));
    
        if isempty(next_stop_idx)
            continue;
        end
    
        num_ok_in_range = sum(rising_idx_ok >= start_idx & rising_idx_ok <= next_stop_idx);
        % 조건z
        if (num_ok_in_range == 4) || (num_ok_in_range == 12)
            valid_pairs = [valid_pairs; start_idx, next_stop_idx];  % 행렬에 쌍 저장
        end
    end

    % === idx_range 자동 설정: 첫 번째 valid_pair 구간 내 OK falling~rising ===
    start_idx = valid_pairs(1,1);
    stop_idx  = valid_pairs(1,2);
    
    ok_fallings = find(falling_edge_ok);
    ok_risings  = find(rising_edge_ok);
    
    % 구간 내 OK falling/rising 찾기
    fall_in_range = ok_fallings(ok_fallings > start_idx & ok_fallings < stop_idx);
    rise_in_range = ok_risings(ok_risings > start_idx & ok_risings < stop_idx);
    
    if numel(fall_in_range) < 1 || numel(rise_in_range) < 2
        error('Not enough OK events found within valid range');
    end
    
    idx_range = fall_in_range(1):rise_in_range(2);   % OK Falling ~ 다음 OK Rising
    
    %% === 유효한 쌍에 대해 Cropping 및 저장 여부 확인 ===
    cropped_data = {};
    for count = 1:size(valid_pairs, 1)
        start_idx = valid_pairs(count, 1);
        stop_idx  = valid_pairs(count, 2);

        % ─── 세그먼트별 퀘터니언 전처리 ──────────────────────────────
        % 1) IMU 쿼터니언 정규화 (각 행 [w x y z])
        q_rthigh_raw = quatnormalize(data_imu1);  
        q_rshank_raw = quatnormalize(data_imu2);  
        q_lthigh_raw = quatnormalize(data_imu3);
        q_lshank_raw = quatnormalize(data_imu4);
        q_pelvic_raw = quatnormalize(data_imu5);
        q_upbody_raw = quatnormalize(data_imu6);
        q_ruparm_raw = quatnormalize(data_imu7);
        q_rfoarm_raw = quatnormalize(data_imu8);
        q_luparm_raw = quatnormalize(data_imu9);
        q_lfoarm_raw = quatnormalize(data_imu10);

        % 2) 모델 Standing‐pose 보정용 쿼터니언 정의
        showplot = false;
        stand_pose_time = start_idx;
        % [~, qD.rthigh] = plotQuatNearestAxes(q_rthigh_raw(stand_pose_time,:), showplot);
        % [~, qD.rshank] = plotQuatNearestAxes(q_rshank_raw(stand_pose_time,:), showplot);
        % [~, qD.lthigh] = plotQuatNearestAxes(q_lthigh_raw(stand_pose_time,:), showplot);
        % [~, qD.lshank] = plotQuatNearestAxes(q_lshank_raw(stand_pose_time,:), showplot);
        % [~, qD.pelvic] = plotQuatNearestAxes(q_pelvic_raw(stand_pose_time,:), showplot);
        % [~, qD.upbody] = plotQuatNearestAxes(q_upbody_raw(stand_pose_time,:), showplot);
        % [~, qD.ruparm] = plotQuatNearestAxes(q_ruparm_raw(stand_pose_time,:), showplot);
        % [~, qD.rfoarm] = plotQuatNearestAxes(q_rfoarm_raw(stand_pose_time,:), showplot);
        % [~, qD.luparm] = plotQuatNearestAxes(q_luparm_raw(stand_pose_time,:), showplot);
        % [~, qD.lfoarm] = plotQuatNearestAxes(q_lfoarm_raw(stand_pose_time,:), showplot);
        qD.rthigh = [1 0 0 0];
        qD.rshank = [1 0 0 0];
        qD.lthigh = [1 0 0 0];
        qD.lshank = [1 0 0 0];
        qD.pelvic = [1 0 0 0];
        qD.upbody = [1 0 0 0];
        qD.ruparm = [1 0 0 0];
        qD.rfoarm = [1 0 0 0];
        qD.luparm = [1 0 0 0];
        qD.lfoarm = [1 0 0 0];
        
        % 3) 첫 프레임(raw 1) 추출 및 고정 보정 상수 계산
        qCorr.rthigh = quatmultiply(qD.rthigh, quatinv(q_rthigh_raw(stand_pose_time,:)));
        qCorr.rshank = quatmultiply(qD.rshank, quatinv(q_rshank_raw(stand_pose_time,:)));
        qCorr.lthigh = quatmultiply(qD.lthigh, quatinv(q_lthigh_raw(stand_pose_time,:)));
        qCorr.lshank = quatmultiply(qD.lshank, quatinv(q_lshank_raw(stand_pose_time,:)));
        qCorr.pelvic = quatmultiply(qD.pelvic, quatinv(q_pelvic_raw(stand_pose_time,:)));
        qCorr.upbody = quatmultiply(qD.upbody, quatinv(q_upbody_raw(stand_pose_time,:)));
        qCorr.ruparm = quatmultiply(qD.ruparm, quatinv(q_ruparm_raw(stand_pose_time,:)));
        qCorr.rfoarm = quatmultiply(qD.rfoarm, quatinv(q_rfoarm_raw(stand_pose_time,:)));
        qCorr.luparm = quatmultiply(qD.luparm, quatinv(q_luparm_raw(stand_pose_time,:)));
        qCorr.lfoarm = quatmultiply(qD.lfoarm, quatinv(q_lfoarm_raw(stand_pose_time,:)));

        % 4) 모든 시점에 보정 곱하기 → 정렬된 쿼터니언

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


        % 역방향이면 모두 반전
        stand_region = start_idx+40 : min(start_idx+50, size(q_upbody,1));

        yaw_all = quat2eul(q_upbody, 'ZYX');        % Yaw-Pitch-Roll 순서
        mean_yaw = mean(wrapToPi( yaw_all(stand_region,1) ));
        invert_flag = abs(mean_yaw) > pi/2;        % 뒤로 보고 있으면 true

        % 3) 초기 정렬 (stand_region: segment 시작 ~ +10샘플)
        % q_upbody = quatmultiply(q_upbody, quatconj(mean(q_upbody(stand_region,:))) );
        % q_ruparm = quatmultiply(q_ruparm, quatconj(mean(q_ruparm(stand_region,:))));
        % q_luparm = quatmultiply(q_luparm, quatconj(mean(q_luparm(stand_region,:))));
        % q_rfoarm = quatmultiply(q_rfoarm, quatconj(mean(q_rfoarm(stand_region,:))));
        % q_lfoarm = quatmultiply(q_lfoarm, quatconj(mean(q_lfoarm(stand_region,:))));

        % 4) 이 세그먼트 내 idx_range 다시 계산 (OK falling→OK rising)
        ok_fallings = find(falling_edge_ok);
        ok_risings  = find(rising_edge_ok);
        fall_in_range = ok_fallings(ok_fallings > start_idx & ok_fallings < stop_idx);
        rise_in_range = ok_risings(ok_risings > start_idx & ok_risings < stop_idx);
        if numel(fall_in_range) < 1 || numel(rise_in_range) < 2
            error('Segment %d 내 OK 이벤트가 부족합니다.', count);
        end
        idx_range = fall_in_range(1) : rise_in_range(2);

        % 5) 좌/우 대칭 최적화
        % [q_luparm, ~] = optimize_orientation(q_luparm, q_upbody, q_ruparm, idx_range);
        % [q_lfoarm, ~] = optimize_orientation(q_lfoarm, q_upbody, q_rfoarm, idx_range);

        % 7) 두 세그먼트 간 상대 퀘터니언 → Euler
        q_rshoulder  = quatmultiply(quatconj(q_upbody),  q_ruparm);
        q_lshoulder  = quatmultiply(quatconj(q_upbody),  q_luparm);
        q_relbow = quatmultiply(quatconj(q_ruparm), q_rfoarm);
        q_lelbow = quatmultiply(quatconj(q_luparm), q_lfoarm);
        q_rhip  = quatmultiply(quatconj(q_pelvic),  q_rthigh);
        q_lhip  = quatmultiply(quatconj(q_pelvic),  q_lthigh);
        q_rknee = quatmultiply(quatconj(q_rthigh), q_rshank);
        q_lknee = quatmultiply(quatconj(q_lthigh), q_lshank);

        euler_rhip       = -rad2deg(quat2eul(q_rhip));
        euler_lhip       =  rad2deg(quat2eul(q_lhip));
        euler_rknee      =  rad2deg(quat2eul(q_rknee));
        euler_lknee      = -rad2deg(quat2eul(q_lknee));
        euler_trunk      =  rad2deg(quat2eul(q_upbody));
        euler_rshoulder  =  rad2deg(quat2eul(q_rshoulder));
        euler_lshoulder  = -rad2deg(quat2eul(q_lshoulder));
        euler_relbow     = -rad2deg(quat2eul(q_relbow));
        euler_lelbow     =  rad2deg(quat2eul(q_lelbow));
        

        if invert_flag
            euler_rshoulder  = -euler_rshoulder;
            euler_lshoulder  = -euler_lshoulder;
            euler_relbow = -euler_relbow;
            euler_lelbow = -euler_lelbow;
            euler_rhip  = -euler_rhip;
            euler_lhip  = -euler_lhip;
            euler_rknee = -euler_rknee;
            euler_lknee = -euler_lknee;
        end

    
        % --- 데이터 자르기 ---
        cropped.time  = data_time(start_idx:stop_idx,1);
        cropped.imu1  = data_imu1(start_idx:stop_idx,:);
        cropped.imu2  = data_imu2(start_idx:stop_idx,:);
        cropped.imu3  = data_imu3(start_idx:stop_idx,:);
        cropped.imu4  = data_imu4(start_idx:stop_idx,:);
        cropped.imu5  = data_imu5(start_idx:stop_idx,:);
        cropped.imu6  = data_imu6(start_idx:stop_idx,:);
        cropped.imu7  = data_imu7(start_idx:stop_idx,:);
        cropped.imu8  = data_imu8(start_idx:stop_idx,:);
        cropped.imu9  = data_imu9(start_idx:stop_idx,:);
        cropped.imu10 = data_imu10(start_idx:stop_idx,:);
        
        
        cropped.trunk = euler_trunk(start_idx:stop_idx,:);
        cropped.rshoulder = euler_rshoulder(start_idx:stop_idx,:);
        cropped.lshoulder = euler_lshoulder(start_idx:stop_idx,:);
        cropped.relbow = euler_relbow(start_idx:stop_idx,:);
        cropped.lelbow = euler_lelbow(start_idx:stop_idx,:);
        cropped.rhip = euler_rhip(start_idx:stop_idx,:);
        cropped.lhip = euler_lhip(start_idx:stop_idx,:);
        cropped.rknee = euler_rknee(start_idx:stop_idx,:);
        cropped.lknee = euler_lknee(start_idx:stop_idx,:);
    
        cropped.emgL1 = data_emgL1(start_idx:stop_idx,1);
        cropped.emgL2 = data_emgL2(start_idx:stop_idx,1);
        cropped.emgL3 = data_emgL3(start_idx:stop_idx,1);
        cropped.emgL4 = data_emgL4(start_idx:stop_idx,1);
        cropped.emgR1 = data_emgR1(start_idx:stop_idx,1);
        cropped.emgR2 = data_emgR2(start_idx:stop_idx,1);
        cropped.emgR3 = data_emgR3(start_idx:stop_idx,1);
        cropped.emgR4 = data_emgR4(start_idx:stop_idx,1);
    
        cropped.rising_ok  = rising_edge_ok(start_idx:stop_idx,1);
        cropped.button_a   = rising_edge_a(start_idx:stop_idx,1);
        cropped.button_b   = rising_edge_b(start_idx:stop_idx,1);
        cropped.falling_ok = falling_edge_ok(start_idx:stop_idx,1);

        if manual_check
            % --- 시각화 ---
            imuupperfig = figure('Position',[10 10 400 1000]);
            subplot(10,1,[1 2 3]); hold on;
            plot(cropped.time, cropped.rshoulder(:,1), 'r'); 
            plot(cropped.time, cropped.lshoulder(:,1), 'b');
            plot(cropped.time, 50*cropped.rising_ok, 'k');
            plot(cropped.time, 50*cropped.falling_ok, 'g');
            xlim([-inf inf]); legend(["RShoulder - Roll", "LShoulder - Roll"], 'Location','best'); grid on
            subplot(10,1,4); hold on;
            plot(cropped.time, cropped.rshoulder(:,2), 'm'); 
            plot(cropped.time, cropped.lshoulder(:,2), 'c'); 
            xlim([-inf inf]); legend(["RShoulder - Pitch", "LShoulder - Pitch"], 'Location','best'); grid on
            subplot(10,1,5); hold on;
            plot(cropped.time, cropped.rshoulder(:,3), 'm'); 
            plot(cropped.time, cropped.lshoulder(:,3), 'c'); 
            xlim([-inf inf]); legend(["RShoulder - Yaw", "LShoulder - Yaw"], 'Location','best'); grid on
            subplot(10,1,[6 7 8]); hold on;
            plot(cropped.time, 20*cropped.rising_ok, 'k');
            plot(cropped.time, 20*cropped.falling_ok, 'g');
            plot(cropped.time, cropped.relbow(:,1), 'r'); 
            plot(cropped.time, cropped.lelbow(:,1), 'b'); 
            xlim([-inf inf]); legend(["RElbow - Roll", "LElbow - Roll"], 'Location','best'); grid on
            subplot(10,1,9); hold on;
            plot(cropped.time, cropped.relbow(:,2), 'm'); 
            plot(cropped.time, cropped.lelbow(:,2), 'c'); 
            xlim([-inf inf]); legend(["RElbow - Pitch", "LElbow - Pitch"], 'Location','best'); grid on
            subplot(10,1,10); hold on;
            plot(cropped.time, cropped.relbow(:,3), 'm'); 
            plot(cropped.time, cropped.lelbow(:,3), 'c'); 
            xlim([-inf inf]); legend(["RElbow - Yaw", "LElbow - Yaw"], 'Location','best'); grid on
            xlabel("Time[msec]");
    

            imulowerfig = figure('Position',[410 10 400 1000]);
            subplot(10,1,[1 2 3]); hold on;
            plot(cropped.time, cropped.rhip(:,1), 'r'); 
            plot(cropped.time, cropped.lhip(:,1), 'b'); 
            plot(cropped.time, 100*cropped.rising_ok, 'k');
            plot(cropped.time, 100*cropped.falling_ok, 'g');
            xlim([-inf inf]); legend(["RHip - Roll", "LHip - Roll"], 'Location','best'); grid on
            subplot(10,1,4); hold on;
            plot(cropped.time, cropped.rhip(:,2), 'm'); 
            plot(cropped.time, cropped.lhip(:,2), 'c'); 
            xlim([-inf inf]); legend(["RHip - Pitch", "LHip - Pitch"], 'Location','best'); grid on
            subplot(10,1,5); hold on;
            plot(cropped.time, cropped.rhip(:,3), 'm'); 
            plot(cropped.time, cropped.lhip(:,3), 'c'); 
            xlim([-inf inf]); legend(["RHip - Yaw", "LHip - Yaw"], 'Location','best'); grid on
            subplot(10,1,[6 7 8]); hold on;
            plot(cropped.time, cropped.rknee(:,1), 'r'); 
            plot(cropped.time, cropped.lknee(:,1), 'b'); 
            plot(cropped.time, 100*cropped.rising_ok, 'k');
            plot(cropped.time, 100*cropped.falling_ok, 'g');
            xlim([-inf inf]); legend(["RKnee - Roll", "LKnee - Roll"], 'Location','best'); grid on
            subplot(10,1,9); hold on;
            plot(cropped.time, cropped.rknee(:,2), 'm'); 
            plot(cropped.time, cropped.lknee(:,2), 'c'); 
            xlim([-inf inf]); legend(["RKnee - Pitch", "LKnee - Pitch"], 'Location','best'); grid on
            subplot(10,1,10); hold on;
            plot(cropped.time, cropped.rknee(:,3), 'm'); 
            plot(cropped.time, cropped.lknee(:,3), 'c'); 
            xlim([-inf inf]); legend(["RKnee - Yaw", "LKnee - Yaw"], 'Location','best'); grid on
            xlabel("Time[msec]");
    
            % labelfig = figure('Position',[10 1050 800 100]); hold on;
            % plot(cropped.time, cropped.rising_ok, 'k', 'LineWidth', 2); xlim([-inf inf]);
            % plot(cropped.time, cropped.falling_ok, 'g', 'LineWidth', 2); xlim([-inf inf]);
            % plot(cropped.time, cropped.button_a,  'b', 'LineWidth', 2); xlim([-inf inf]);
            % plot(cropped.time, cropped.button_b,  'r', 'LineWidth', 2); xlim([-inf inf]);
            % legend('Labeling', 'Start', 'Stop');
            % title("Button Labeling (Check 4 black lines)"); xlabel("Time[msec]");
    
            % --- 저장 여부 질의 ---
            while true
                user_input = input('Do you want to save? (y/n): ', 's');
                if strcmpi(user_input, 'y')
                    cropped_data{end+1} = cropped;
                    disp('Data Appended.');
                    close all
                    break;
                elseif strcmpi(user_input, 'n')
                    disp('Data Rejected.');
                    close all
                    break;
                elseif strcmpi(user_input, 'skip')
                    disp('Skipping this trial set.')
                    close all
                    return;
                else
                    disp('Wrong input.');
                end
            end

        else
            cropped_data{end+1} = cropped;
        end
    end
    
    disp('Finish appending.');
    
    %%
    cropped_filename = strrep(filename, 'sensor_', 'cropped_');
    for i = 1:length(cropped_data)
        trial = cropped_data{i};
        group_prefix = sprintf('/trial_%d', i);
        
        % 저장할 필드 리스트
        fields = fieldnames(trial);
        
        for j = 1:numel(fields)
            name = fields{j};
            path = sprintf('%s/%s', group_prefix, name);
            data = trial.(name);
    
            % h5create는 처음만 실행 (존재하면 skip 또는 try-catch)
            if ~isfile(cropped_filename) || ~h5exists(cropped_filename, path)
                h5create(cropped_filename, path, size(data), 'Datatype', class(data));
            end
    
            % 데이터 저장
            h5write(cropped_filename, path, data);
        end
    end
    
    disp('File created.');

end
