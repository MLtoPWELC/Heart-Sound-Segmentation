function [delta, psi, qt] = viterbiDecodePCG_Springer(signal, observation_probs, pi_vector, heartrate, systolic_time, Fs,figures,alpha)

if nargin < 5
    figures = false;
end

%% Preliminary

T = length(observation_probs);
N = 4; % Number of state s

max_duration_D = round((1*(60/heartrate))*Fs);

delta = ones(T+ max_duration_D-1,N)*-inf;

psi = zeros(T+ max_duration_D-1,N);

psi_duration =zeros(T + max_duration_D-1,N);

%% Setting up state duration probabilities, using Gaussian distributions:
[d_distributions, max_S1, min_S1, max_S2, min_S2, max_systole, min_systole, max_diastole, min_diastole] = get_duration_distributions(heartrate,systolic_time);

duration_probs = zeros(N,max_duration_D);
duration_sum = zeros(N,1);
for state_j = 1:N
    for d = 1:max_duration_D
        if(state_j == 1)
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            if(d < min_S1 || d > max_S1)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif(state_j==3)
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            if(d < min_S2 || d > max_S2)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif(state_j==2)
            
            duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),cell2mat(d_distributions(state_j,2)));
            if(d < min_systole|| d > max_systole)
                duration_probs(state_j,d)= realmin;
            end
            
            
        elseif (state_j==4)
%change the model of diastole
%             duration_probs(state_j,d) = mvnpdf(d,cell2mat(d_distributions(state_j,1)),2*cell2mat(d_distributions(state_j,2)));
            duration_probs(state_j,d) = poisspdf(d,cell2mat(d_distributions(state_j,1)));
             
%             if(d < min_diastole ||d > max_diastole)
%                 duration_probs(state_j,d)= realmin;
%             end

            
        end
    end
    duration_sum(state_j) = sum(duration_probs(state_j,:));
end


if(length(duration_probs)>max_duration_D)
    duration_probs(:,(max_duration_D+1):end) = [];
end

if(figures)
    figure('Name', 'Duration probabilities');
    plot(duration_probs(1,:)./ duration_sum(1),'Linewidth',2);
    hold on;
    plot(duration_probs(2,:)./ duration_sum(2),'r','Linewidth',2);
    hold on;
    plot(duration_probs(3,:)./ duration_sum(3),'g','Linewidth',2);
    hold on;
    plot(duration_probs(4,:)./ duration_sum(4),'k','Linewidth',2);
    hold on;
    legend('S1 Duration','Systolic Duration','S2 Duration','Diastolic Duration');
%     pause();
end
%% Perform the actual Viterbi Recursion:

qt = zeros(1,length(delta));
%% Initialisation Step

delta(1,:) = log(pi_vector) + log(observation_probs(1,:)); 

psi(1,:) = -1;

a_matrix = [0,1,0,0;0 0 1 0; 0 0 0 1;1 0 0 0];


%% The Viterbi algorithm:

for t = 2:T+ max_duration_D-1
    for j = 1:N
        for d = 1:1:max_duration_D
           
            start_t = t - d;
            if(start_t<1)
                start_t = 1;
            end
            if(start_t > T-1)
                start_t = T-1;
            end
            
            end_t = t;
            if(t>T)
                end_t = T;
            end
            
            [max_delta, max_index] = max(delta(start_t,:)+log(a_matrix(:,j))');
          
            probs = prod(observation_probs(start_t:end_t,j));
            
            
            if(probs ==0)
                probs = realmin;
            end
            emission_probs = log(probs);
            
            
            if(emission_probs == 0 || isnan(emission_probs))
                emission_probs =realmin;
            end
            
            delta_temp = max_delta + (emission_probs)+ alpha*log((duration_probs(j,d)./duration_sum(j))); 
            % the core equation 
         
            if(delta_temp>delta(t,j))
                delta(t,j) = delta_temp;
                psi(t,j) = max_index;
                psi_duration(t,j) = d;
            end
            
        end
    end
end



%% Termination

temp_delta = delta(T+1:end,:);
[~, idx] = max(temp_delta(:));
[pos, ~] = ind2sub(size(temp_delta), idx);

pos = pos+T;


%1)
[~, state] = max(delta(pos,:),[],2);

%2)
offset = pos;
preceding_state = psi(offset,state);

%3)
% state_duration = psi_duration(offset, state);
onset = offset - psi_duration(offset,state)+1;

%4)
qt(onset:offset) = state;

state = preceding_state;

count = 0;

while(onset > 2)
    
    %2)
    offset = onset-1;
    %     offset_array(offset,1) = inf;
    preceding_state = psi(offset,state);
    %     offset_array(offset,2) = preceding_state;
    
    
    %3)
    %     state_duration = psi_duration(offset, state);
    onset = offset - psi_duration(offset,state)+1;
    
    %4)
    %     offset_array(onset:offset,3) = state;
    
    if(onset<2)
        onset = 1;
    end
    qt(onset:offset) = state;
    state = preceding_state;
    count = count +1;
    
    if(count> 1000)
        break;
    end
end

qt = qt(1:T);


