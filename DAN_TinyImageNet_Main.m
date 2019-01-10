
% Title   : "Stacking-based Deep Neural Network: Deep Analytic Network for Pattern Classification" 
% Authors : C. Y. Low*, J. Park, A. B. J. Teoh 
% Affl.   : Multimedia University, Malaysia*, Yonsei University, Seoul, South Korea;  
% Email   : chengyawlow@gmail.com; julypraise@gmail.com; bjteoh@yonsei.ac.kr
% URL     : https://arxiv.org/abs/1811.07184

% Deep Analytic Networks on Pre-trained VD-VGG-19 and ResNet-156 Features using TinyImageNet
% 1.  Download the pre-learned, pre-extracted VD-VGG-19 & ResNet-156 features
%     (a) Pre-learned, pre-extracted VD-VGG-19 : https://drive.google.com/open?id=1yqflW8M_e5_tPxELu6HEPpyz3J1RWqBx
%     (b) Pre-learned, pre-extracted ResNet-156 : https://drive.google.com/open?id=1N8w81BoV38Ub8fjQ_HPJAhIDpIqhXoAM
% 2.  Run DAN_TinyImageNet_Main

% Set FEA_ID
% FEA_ID = 1 : VD_VGG
% FEA_ID = 2 : ResNet
% FEA_ID = 3 : VD_VGG + ResNet

function DAN_TinyImageNet_Main( FEA_ID )
 
    clc;
    fprintf( '\n' );
    fprintf( ' -----------------------------------------------------------\n' );
    fprintf( '                    DAN_TinyImageNet_Main                   \n' );
    fprintf( ' -----------------------------------------------------------\n' );    
    
    %% Configure DAN Parameters
    % Set FEA_ID, if nargin == 0
    if nargin == 0
        FEA_ID = 3;
    end
   
    NUM_LAYER = 8;
    
    % Set NET_DESCR, with respect to FEA_ID
    if FEA_ID == 1
        NET_DESCR = 'DAN + VD-VGG';
    elseif FEA_ID == 2
        NET_DESCR = 'DAN + ResNet';
    elseif FEA_ID == 3
        NET_DESCR = 'DAN + VD-VGG + ResNet';
    end
    
    % Set Class Indicator Matrix ( CIM )
    CIM = [ 0, 1 ];
    
    % Set LAMBDA & LAMBDA_FT 
    LAMBDA = 10;
    LAMBDA_FT = 0.1;
    LAMBDA = LAMBDA .* ones( 1, NUM_LAYER );
    LAMBDA_FT = LAMBDA_FT .* ones( 1, NUM_LAYER );
    
    % Set element-wise POW_FT, i.e., power-law norm. ratio
    POW_FT = 1;
        
    % Set NL
    % NL = 0 : Linear
    % NL = 1 : ReLU
    NL = 1; 
    if NL == 0
        NL_DESCR = 'Linear';
    elseif NL == 1
        NL_DESCR = 'ReLU';
    end
      
    %% Summarize & Display DAN/KDAN Parameters
    DAN.NUM_LAYER = NUM_LAYER;
    DAN.NET_DESCR = NET_DESCR;    
    DAN.CIM = CIM;
    DAN.LAMBDA = LAMBDA;
    DAN.LAMBDA_FT = LAMBDA_FT;
    DAN.POW_FT = POW_FT;
    DAN.NL = NL;
    DAN.NL_DESCR = NL_DESCR;
    
    % Display DAN Parameters
    DAN
    
    %% Load VD_VGG Features     
    % Initialize X_DAN_TR_ALL & X_DAN_TT_ALL
    X_DAN_TR_ALL = [];
    X_DAN_TT_ALL = [];
        
    % Load pre-learned VD_VGG features, with respect to FEA_ID
    if FEA_ID == 1 || FEA_ID == 3
    
        fprintf( '\n' );
        fprintf( '**********' );   
        fprintf( '\n' );
        
        fprintf( '\n' );
        fprintf( 'LOADING PRE-LEARNED VD_VGG FEATURES ...' );
        fprintf( '\n' );
    
        % Load X_DAN_TR, Y_DAN_TR, X_DAN_TT, Y_DAN_TT
        load( 'VD_VGG_TinyImageNet.mat', 'X_DAN_TR', 'Y_DAN_TR', 'X_DAN_TT', 'Y_DAN_TT' );
   
        X_DAN_TR_ALL = cat( 1, X_DAN_TR_ALL, X_DAN_TR );
        X_DAN_TT_ALL = cat( 1, X_DAN_TT_ALL, X_DAN_TT );
    
        clear X_DAN_TR X_DAN_TT;
        pause( 0.1 );
        
    end
    
    % Load pre-learned ResNet features, with respect to FEA_ID
    if FEA_ID == 2 || FEA_ID == 3
        
        % fprintf( '\n' );
        % fprintf( '**********' );   
        % fprintf( '\n' );
        
        fprintf( '\n' );
        fprintf( 'LOADING PRE-LEARNED ResNet FEATURES ...' );
        fprintf( '\n' );
        
        % Load X_DAN_TR, Y_DAN_TR, X_DAN_TT, Y_DAN_TT
        load( 'ResNet_TinyImageNet.mat', 'X_DAN_TR', 'Y_DAN_TR', 'X_DAN_TT', 'Y_DAN_TT' );
   
        X_DAN_TR_ALL = cat( 1, X_DAN_TR_ALL, X_DAN_TR );
        X_DAN_TT_ALL = cat( 1, X_DAN_TT_ALL, X_DAN_TT );
    
        clear X_DAN_TR X_DAN_TT;
        pause( 0.1 );
        
    end
    
    X_DAN_TR = X_DAN_TR_ALL;
    X_DAN_TT = X_DAN_TT_ALL;
    clear X_DAN_TR_ALL X_DAN_TT_ALL;
    pause( 0.0001 );
    
    % Display X_DAN_TR_SZ & Y_DAN_TR_SZ
    X_DAN_TR_SZ = size( X_DAN_TR )
    Y_DAN_TR_SZ = size( Y_DAN_TR )
    
    % Display X_DAN_TT_SZ & Y_DAN_TT_SZ
    X_DAN_TT_SZ = size( X_DAN_TT )
    Y_DAN_TT_SZ = size( Y_DAN_TT )
    
    %% Set Other DAN Parameters
    fprintf( '\n' );
    fprintf( '**********' );   
    fprintf( '\n' );
    % fprintf( '\n' );
    
    Y_DAN_UNIQ = unique( cat( 1, Y_DAN_TR, Y_DAN_TT ) );
    
    % Set other parameters
    NUM_CLS = numel( Y_DAN_UNIQ );
    NUM_DIM = X_DAN_TR_SZ( 1 );
    
    assert( min( Y_DAN_TR ) == 1 && max( Y_DAN_TR ) == NUM_CLS );
    assert( min( Y_DAN_TT ) == 1 && max( Y_DAN_TT ) == NUM_CLS );
    pause( 0.0001 );
    
    % #################################################################
    % Define one-hot encoded CIM for Y_DAN_TR, i.e., Y_DAN_CIM_TR
    % Y_DAN_TR_CIM : ROW
    Y_DAN_TR_UNIQ = unique( Y_DAN_TR );
    
    Y_DAN_CIM_TR = zeros( numel( Y_DAN_TR ), numel( Y_DAN_TR_UNIQ ) );
    for UNIQ_ID = 1 : numel( Y_DAN_TR_UNIQ )
        Y_DAN_CIM_TR( Y_DAN_TR == Y_DAN_TR_UNIQ( UNIQ_ID ), UNIQ_ID ) = 1;   
    end
    Y_DAN_CIM_TR( Y_DAN_CIM_TR == 0 ) = DAN.CIM( 1 );
    Y_DAN_CIM_TR( Y_DAN_CIM_TR == 1 ) = DAN.CIM( 2 );

    % Centralize Y_DAN_CIM_TR by ROW
    Y_DAN_CIM_TR = bsxfun( @minus, Y_DAN_CIM_TR, mean( Y_DAN_CIM_TR, 2 ) );
       
    Y_DAN_CIM_TR_MAX = max( Y_DAN_CIM_TR( : ) )
    Y_DAN_CIM_TR_MIN = min( Y_DAN_CIM_TR( : ) )
    
    %% Learning DAN from X_DAN_TR, with respect to pre-set DAN parameters  
    % fprintf( '\n' );
    % fprintf( '**********' );   
    % fprintf( '\n' );
    
    fprintf( '\n' );
    fprintf( '\n' );
    fprintf(' ~~~~~ LEARNING AN FROM X_DAN_TR ~~~~~ ');
    fprintf( '\n' );
    
    % Initialize relevant parameters       
    recogRate_FT_TR = zeros( 1, NUM_LAYER );  
    recogRate_FT_TT = zeros( 1, NUM_LAYER ); 
  
    DAN_RR_W_ALL = cell( 1, NUM_LAYER );   
    DAN_RR_W_FT_ALL = cell( 1, NUM_LAYER ); 
        
    % Initialize X_DAN_TR & X_DAN_TR_IN
    % X_DAN_TR & X_DAN_TR_IN : COLUMNAR
    % X_DAN_TR = [];
    X_DAN_FT_TR = [];
    
    for LAYER_ID = 1 : NUM_LAYER  
        
        fprintf( '\n' );
        fprintf( 'LEARNING DAN/K-DAN FOR LAYER ID : %d  ... ', LAYER_ID );
        fprintf( '\n' );
        
        % Instead of multiplying by the inverse, use matrix right division (/) or matrix left division (\). 
        % Replace inv(A)*b with A\b
        % Replace b*inv(A) with b/A
        % Perform RR in primal or dual form, accordingly
        RR_TYPE = 1;
        if size( X_DAN_TR, 1 ) > size( X_DAN_TR, 2 )
            RR_TYPE = 2;
        end

        % Perform RR in primal form
        if RR_TYPE == 1
            DAN_RR_W_TEMP = ( ( X_DAN_TR * X_DAN_TR' ) + ( LAMBDA( LAYER_ID ) * eye( size( X_DAN_TR, 1 ) ) ) ) \ ( X_DAN_TR * Y_DAN_CIM_TR ); 
        % Perform RR in dual form
        elseif RR_TYPE == 2
            DAN_RR_W_TEMP = ( ( X_DAN_TR' * X_DAN_TR ) + ( LAMBDA( LAYER_ID ) * eye( size( X_DAN_TR, 2 ) ) ) ) \ Y_DAN_CIM_TR; 
            DAN_RR_W_TEMP = X_DAN_TR * DAN_RR_W_TEMP;
        end
        
        DAN_RR_W_ALL{ LAYER_ID } = DAN_RR_W_TEMP;  
        clear DAN_RR_W_TEMP;
        pause(0.001);

        % DAN_PRED_TR : COLUMNAR
        DAN_PRED_TR = DAN_RR_W_ALL{ LAYER_ID }' * X_DAN_TR; 
       
        % Apply non-linearity, based on NL
        if NL == 1
            DAN_PRED_TR = max( 0, DAN_PRED_TR );
            DAN_PRED_TR = min( 1, DAN_PRED_TR );
        end

        % Extend X_DAN_TR
        % X_DAN_TR : COLUMNAR
        % X_DAN_TR = X_DAN_TR + DAN_PRED_TR
        X_DAN_TR = cat( 1, X_DAN_TR, DAN_PRED_TR );
            
        %% DAN Fine-Tuning Layer 
        % X_DAN_FT_TR : COLUMNAR
        % X_DAN_FT_TR includes only power-normalized DAN_PRED_TR for ALL layers
        X_DAN_FT_TR_TEMP = sign( DAN_PRED_TR ) .* ( abs( DAN_PRED_TR ) .^ POW_FT );
        X_DAN_FT_TR = cat( 1, X_DAN_FT_TR, X_DAN_FT_TR_TEMP );

        clear DAN_PRED_TR X_DAN_FT_TR_TEMP;
        pause( 0.0001 );

        % Perform RR_FT in primal or dual form, accordingly
        RR_FT_TYPE = 1;
        if size( X_DAN_FT_TR, 1 ) > size( X_DAN_FT_TR, 2 )
           RR_FT_TYPE = 2;
        end

        % Perform RR_FT in primal form
        if RR_FT_TYPE == 1
           DAN_RR_W_FT_TEMP = ( ( X_DAN_FT_TR * X_DAN_FT_TR' ) + ( LAMBDA_FT( LAYER_ID ) * eye( size( X_DAN_FT_TR, 1 ) ) ) ) \ ( X_DAN_FT_TR * Y_DAN_CIM_TR ); 
           DAN_RR_W_FT_ALL{ LAYER_ID } = DAN_RR_W_FT_TEMP;   
        % Perform RR_FT in dual form
        elseif RR_FT_TYPE == 2
            % DAN_RR_W_FT_ALL{LAYER_ID} : DIM x CLS
            DAN_RR_W_FT_TEMP = ( ( X_DAN_FT_TR' * X_DAN_FT_TR ) + ( LAMBDA_FT( LAYER_ID ) * eye( size( X_DAN_FT_TR, 2 ) ) ) ) \ Y_DAN_CIM_TR; 
            DAN_RR_W_FT_ALL{ LAYER_ID } = X_DAN_FT_TR * DAN_RR_W_FT_TEMP;  
        end
        clear DAN_RR_W_FT_TEMP;
        pause( 0.0001 );

        DAN_PRED_FT_TR = DAN_RR_W_FT_ALL{ LAYER_ID }' * X_DAN_FT_TR; 

        % Evaluate DAN performance w/ FT, with respect to DAN_PRED_FT_TR 
        recogRate_FT_TR( LAYER_ID ) = estimate_recogRate_TR( DAN_PRED_FT_TR, Y_DAN_TR )

        clear DAN_PRED_FT_TR;
        pause( 0.0001 );

    end

    %% Applying DAN to X_DAN_TT, with respect to pre-learned DAN     
    % fprintf( '\n' );
    % fprintf( '**********' );   
    % fprintf( '\n' );
    
    fprintf( '\n' );
    fprintf( ' ~~~~~ APPLYING DAN/K-DAN TO X_DAN_TT ~~~~~ ' );
    fprintf( '\n' );
    
    % Initialize X_DAN_TT & X_DAN_FT_TT
    % X_DAN_TT & X_DAN_FT_TT : COLUMNAR
    % X_DAN_TT = [];
    X_DAN_FT_TT = [];

    for LAYER_ID = 1 : NUM_LAYER 
        
        fprintf( '\n' );
        fprintf( 'APPLYING DAN/K-DAN FOR LAYER ID : %d  ... ', LAYER_ID );
        fprintf( '\n' );
        
        % Apply DAN_RR_W_ALL to X_DAN_TT
        DAN_PRED_TT = DAN_RR_W_ALL{ LAYER_ID }' * X_DAN_TT;
        DAN_RR_W_ALL{ LAYER_ID } = [];
        pause( 0.0001 );
        
        % Apply non-linearity, based on NL
        if NL == 1 
            DAN_PRED_TT = max( 0, DAN_PRED_TT );
            DAN_PRED_TT = min( 1, DAN_PRED_TT );
        end
                
        % X_DAN_TT : COLUMNAR
        % X_DAN_TT = X_DAN_TT + DAN_PRED_TT
        X_DAN_TT = cat( 1, X_DAN_TT, DAN_PRED_TT );
   
       %% DAN Fine-Tuning Layer 
        % X_DAN_FT_TT : COLUMNAR
        % X_DAN_FT_TT includes only power-normalized DAN_PRED_TT for ALL layers
        X_DAN_FT_TT_TEMP = sign( DAN_PRED_TT ) .* ( abs( DAN_PRED_TT ) .^ POW_FT );
        X_DAN_FT_TT = cat( 1, X_DAN_FT_TT, X_DAN_FT_TT_TEMP );
        
        clear DAN_PRED_TT X_DAN_FT_TT_TEMP;
        pause( 0.0001 );

        % Apply DAN_RR_W_FT_ALL to X_DAN_FT_TT
        % DAN_PRED_FT_TT : COLUMNAR
        DAN_PRED_FT_TT = DAN_RR_W_FT_ALL{ LAYER_ID }' * X_DAN_FT_TT; 
        DAN_RR_W_FT_ALL{ LAYER_ID } = [];
        pause( 0.0001 );
        
        % Evaluate DAN performance w/ FT, with respect to DAN_PRED_FT_TT 
        recogRate_FT_TT( LAYER_ID ) = estimate_recogRate_TT( DAN_PRED_FT_TT, Y_DAN_TT )
                                  
        clear DAN_PRED_FT_TT;
        pause( 0.0001 );

    end

    % Display recogRate_TT_ALL & recogRate_FT_TT_ALL
    % fprintf( '\n' );
    % fprintf( '**********' );   
    % fprintf( '\n' );
    
    fprintf( '\n' );
    fprintf( '~~~~~ PERFORMANCE SUMMARY ~~~~~' );
    fprintf( '\n' );

    recogRate_FT_TT
    
    % Summarize recogRate_BEST, recogRate_BEST_ID
    [ recogRate_BEST_FT, BEST_LAYER_ID_FT ] = max( recogRate_FT_TT, [], 2 )
    
    %% Display DAN Parameters
    DAN
    
    %% Clear ALL
    clear all;
        
end

%% Calculate recogRate_TR, recogRate_TT
% DAN_PRED_TR & Y_TR : COLUMNAR
function recogRate = estimate_recogRate_TR( DAN_PRED_TR, Y_TR )
        
    [ ~, Y_TR_EST ] = max( DAN_PRED_TR, [], 1 );
       
    Y_TR_ORI = Y_TR;
    Y_TR_EST = Y_TR_EST';
    
    recogRate = sum( Y_TR_ORI == Y_TR_EST ) / numel( Y_TR_ORI ) * 100;
 
    clearvars -except recogRate;
    pause( 0.0001 );
    
end
 
function recogRate = estimate_recogRate_TT( DAN_PRED_TT, Y_TT )
        
    [ ~, Y_TT_EST ] = max( DAN_PRED_TT, [], 1 );
    
    Y_TT_ORI = Y_TT;
    Y_TT_EST = Y_TT_EST';
    
    recogRate = sum( Y_TT_ORI == Y_TT_EST ) / numel( Y_TT_ORI ) * 100;
    
    clearvars -except recogRate;
    pause( 0.0001 );
        
end
