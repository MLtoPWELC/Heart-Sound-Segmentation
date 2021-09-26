function [a_Se,a_P,a_F1]=F1_score(preds_viterbi,index_use,a)


true_L = a;


all_MINE_S1 = [];
all_MINE_S2 = [];
all_TRUE_S1 = [];
all_TRUE_S2 = [];
right_MINE_S1 = [];
right_MINE_S2 = [];

for i = 1:length(preds_viterbi)
    %%
    seg_label = preds_viterbi{i};
    MINE_S1 = [];MINE_S2 = [];
    
    len_seg = numel(seg_label);
  
    if seg_label(1) == 1
        MINE_S1 = 1;
    end
    for M1 = 2:len_seg
        if seg_label(M1) == 1 && seg_label(M1-1) ~= 1
            MINE_S1 = [MINE_S1 M1];
        end
        
        if M1 == len_seg && seg_label(M1) == 1
            MINE_S1(end) = floor((MINE_S1(end)+M1)/2);
            break
        else
            if seg_label(M1) == 1 && seg_label(M1+1) ~= 1
                MINE_S1(end) = floor((MINE_S1(end)+M1)/2);
            end
        end
    end
    if seg_label(1) == 3
        MINE_S2 = 1;
    end
    for M2 = 2:len_seg
        if seg_label(M2) == 3 && seg_label(M2-1) ~= 3
            MINE_S2 = [MINE_S2 M2];
        end
        
        if M2 == len_seg && seg_label(M2) == 3
            MINE_S2(end) = floor((MINE_S2(end)+M2)/2);
            break
        else
            if seg_label(M2) == 3 && seg_label(M2+1) ~= 3
                MINE_S2(end) = floor((MINE_S2(end)+M2)/2);
            end
        end
    end
    %%
    true_label = true_L{i};
    TRUE_S1 = [];TRUE_S2 = [];
    
    len_true = numel(true_label);
    true_label = [true_label true_label(end)*ones(1,4)];
    if true_label(1) == 1
        TRUE_S1 = 1;
    end
    for N1 = 2:len_true
        if true_label(N1) == 1 && true_label(N1-1) ~= 1
            TRUE_S1 = [TRUE_S1 N1];
        end
        
        if N1 == len_true && true_label(N1) == 1
            TRUE_S1(end) = floor((TRUE_S1(end)+N1)/2);
            break
        else
            if true_label(N1) == 1 && true_label(N1+1) ~= 1
                TRUE_S1(end) = floor((TRUE_S1(end)+N1)/2);
            end
        end
        
    end
    if true_label(1) == 3
        TRUE_S2 = 1;
    end
    for N2 = 2:len_true
        if true_label(N2) == 3 && true_label(N2-1) ~= 3
            TRUE_S2 = [TRUE_S2 N2];
        end
        
        if N2 == len_true && true_label(N2) == 3
            TRUE_S2(end) = floor((TRUE_S2(end)+N2)/2);
            break
        else
            if true_label(N2) == 3 && true_label(N2+1) ~= 3
                TRUE_S2(end) = floor((TRUE_S2(end)+N2)/2);
            end
        end
    end
    %%
    all_MINE_S1 = [all_MINE_S1 numel(MINE_S1)];
    all_MINE_S2 = [all_MINE_S2 numel(MINE_S2)];
    all_TRUE_S1 = [all_TRUE_S1 numel(TRUE_S1)];
    all_TRUE_S2 = [all_TRUE_S2 numel(TRUE_S2)];
    range = 4; %because it is ">" not ">="
    num_MINE_r_S1 = 0;
    for j = 1:numel(TRUE_S1)
        num_MINE_r_S1 = num_MINE_r_S1 + (sum((TRUE_S1(j)-range<MINE_S1)&(MINE_S1<TRUE_S1(j)+range))>0);
    end
    num_MINE_r_S2 = 0;
    for j = 1:numel(TRUE_S2)
        num_MINE_r_S2 = num_MINE_r_S2 + (sum((TRUE_S2(j)-range<MINE_S2)&(MINE_S2<TRUE_S2(j)+range))>0);
    end
    
    right_MINE_S1 = [right_MINE_S1 num_MINE_r_S1];
    right_MINE_S2 = [right_MINE_S2 num_MINE_r_S2];

end


sum_MINE = sum(all_MINE_S1)+sum(all_MINE_S2);
sum_TRUE = sum(all_TRUE_S1)+sum(all_TRUE_S2);
sum_MINE_right =(sum(right_MINE_S1)+sum(right_MINE_S2));

a_Se = sum_MINE_right/sum_TRUE;
a_P = sum_MINE_right/sum_MINE;
a_F1 = 2*a_P*a_Se/(a_P+a_Se);