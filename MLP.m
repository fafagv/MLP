% Define the XOR inputs and expected outputs
inputs = [0 0; 0 1; 1 0; 1 1]';
targets = [0; 1; 1; 0]';
wa = [0.34; 0.16; 0.92]; 
wb = [0.12; 0.57; 0.33]; 
wc = [0.99; 0.16; 0.75]; 
eta = 0.5;
sigmoid = @(x) 1 ./ (1 + exp(-x));
iterations = 10000;
for epoch = 1:iterations
    for i = 1:size(inputs, 2)
        x = [1; inputs(:, i)]; 
        oa = sigmoid(wa' * x);
        ob = sigmoid(wb' * x);
        oc_input = [1; oa; ob];
        oc = sigmoid(wc' * oc_input);
        error = targets(i) - oc;
        delta_oc = oc * (1 - oc) * error;
        delta_oa = oa * (1 - oa) * wc(2) * delta_oc;
        delta_ob = ob * (1 - ob) * wc(3) * delta_oc;

        wc(1) = wc(1) + eta * delta_oc * 1; 
        wc(2) = wc(2) + eta * delta_oc * oa;
        wc(3) = wc(3) + eta * delta_oc * ob;

        wa(1) = wa(1) + eta * delta_oa * 1; 
        wa(2) = wa(2) + eta * delta_oa * inputs(1, i);
        wa(3) = wa(3) + eta * delta_oa * inputs(2, i);

        wb(1) = wb(1) + eta * delta_ob * 1; 
        wb(2) = wb(2) + eta * delta_ob * inputs(1, i);
        wb(3) = wb(3) + eta * delta_ob * inputs(2, i);
    end
end

outputs = zeros(1, size(inputs, 2));
for i = 1:size(inputs, 2)
    x = [1; inputs(:, i)];
    oa = sigmoid(wa' * x);
    ob = sigmoid(wb' * x);
    oc_input = [1; oa; ob];
    outputs(i) = sigmoid(wc' * oc_input);
end

roundedOutputs = round(outputs);

disp('Inputs:');
disp(inputs');
disp('Target Outputs:');
disp(targets');
disp('Network Outputs:');
disp(outputs');
disp('Rounded Network Outputs:');
disp(roundedOutputs');
