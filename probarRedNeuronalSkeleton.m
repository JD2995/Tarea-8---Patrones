function probarRedNeuronal    
   pruebaXOR;
end

function probarMNIST
    % Change the filenames if you've saved the files under different names
    % On some platforms, the files might be saved as 
    % train-images.idx3-ubyte / train-labels.idx1-ubyte
    %images = loadMNISTImages('train-images-idx3-ubyte.gz');
    %labels = loadMNISTLabels('train-labels-idx1-ubyte.gz');
    %tomado de http://www.cad.zju.edu.cn/home/dengcai/Data/MLData.html
    %contains variables 'fea', 'gnd', 'trainIdx' and 'testIdx'. Each row of 'fea' is a sample; 'gnd' is the label.
    load('2k2k.mat');
    a = fea(1,:);
    mat = vec2mat(a',28);
    figure; imagesc(mat);
    
end

function pruebaDatosR2
    [X, T] = generarDatosR2;
    %0.05 para pocas neuronas en la capa oculta
    %[2 25 1], 0.25, casi converge
    %inicializarRed([2 10 1], 0.1, 1);
    %ENTRE MAS NEURONAS EN LA CAPA OCULTA, MAS ALTO DEBE SER ALPHA
    redEj1 = inicializarRed([2 2 1], 0.01, 1);
    numIter = 10000;
    redEj1 = entrenarRed(numIter, X, T, redEj1);
end
 
function [X, T] = generarDatosR2
    %GRUPO 1
    mu = [2,10];
    sigma = [1,1.5;1.5,3];
    %rng default  % For reproducibility
    r = mvnrnd(mu,sigma,100);

    %GRUPO 2
    mu2 = [2,7];
    sigma2 = [1,1.5;1.5,3];
    %rng default  % For reproducibility
    r2 = mvnrnd(mu2,sigma2,50);

    figure
    plot(r(:,1),r(:,2),'+')
    hold on;
    plot(r2(:,1),r2(:,2),'*')

    values = [r; r2];
    targeta = zeros(100,1);
    targetb = ones(50,1);
    target = [targeta; targetb];
    X = values;
    T = target;


end

function pruebaXOR
    X = [0 0;1 0;0 1;1 1];
    T = [0 1 1 0];
    redEj1 = inicializarRed([2 2 1], 0.01, 1);
    numIter = 10000;
    redEj1 = entrenarRed(numIter, X, T, redEj1);
end

function y = evaluarMuestra(x, red)
    
    
end

function red = entrenarRed(numIter, X, T, red)
    red = asignarEntrada(X(1,:),red);
    red = asignarSalidaDeseada(T,red);
    red = calcularPasadaAdelanteEnCapa(red,1);
    red = calcularPasadaAdelanteEnCapa(red,2);
end


function numErrores = evaluarClasificacionesErroneas(X, T, red)    
    
   
end


function red = actualizarPesosSegunDeltas(red)
    
end



%Recibe un vector fila y lo asigna a las neuronas de entrada en la red
function red = asignarEntrada(x, red)
   red.X = [1 x];
end


function red = asignarSalidaDeseada(t, red)
    red.T = t;
end

function red = calcularPasadaAtrasEnCapa(red, numCapa)
    

end

function red = calcularPasadaAdelanteEnCapa(red, numCapa)
    %Pesos netos capa oculta
    [~,col] = size(red.W{numCapa});
    for i=1:col
       red.pesoNeto{numCapa}(i) = red.X*red.W{numCapa}(:,i); 
    end
    red.Y{numCapa} = sigmoid(red.pesoNeto{numCapa});
end


%se supondra que la primer capa es la capa de entrada
function red = inicializarRed(numNeuronasPorCapa, alpha, maxPesosRand)
    D = numNeuronasPorCapa(1);
    M = numNeuronasPorCapa(2);
    K = numNeuronasPorCapa(3);
    WO = rand(D+1,M)*maxPesosRand;
    WS = rand(M+1,K)*maxPesosRand;
    red = struct('alpha',[],'W',[],'pesoNeto',[],'X',[],'Y',[],'T',[]);
    red.alpha = alpha;
    red.W = {WO,WS};
    red.pesoNeto = {zeros(M,1), zeros(K,1)};
    red.X = zeros(1,D);
    red.Y = {zeros(M,1),zeros(K,1)};
    red.T = zeros(K,1);
end

function y = sigmoid(x)
    y = 1./ (1 + exp(-x));
end