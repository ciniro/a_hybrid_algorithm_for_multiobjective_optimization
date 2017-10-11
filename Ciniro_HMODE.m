%------------------------------------------------------------------------%
%          XIII Brazilian Congress on Computational Intelligence         %
%                      Niter�i, Rio de Janeiro, Brazil                   %
%                         2017-30-10 to 2017-11-01                       %
%                                                                        %
%                      CINIRO APARECIDO LEITE NAMETALA                   %
%                         (IFMG, Bambu�, MG, Brazil)                     %
%                                                                        %
%                              GISELE L. PAPPA                           %
%                          EDUARDO GONTIJO CARRANO                       %
%                     (UFMG, Belo Horizonte, MG, Brazil)                 %
%------------------------------------------------------------------------%
%                                                                        %
%            EVOLU��O DIFERENCIAL H�BRIDO COM K-MEANS E NSGA II          %
%            A Hybrid Algorithm for Multiobjective Optimization          %
%                        (MODE + K means + NSGA II)                      %
%                                                                        %
%------------------------------------------------------------------------%

function [ind_best, aval_best, IGD_best, IGD_m, ind_pior, aval_pior, IGD_pior] = Ciniro_HMODE(n_aval, problem, n_obj, n_exec)
%
%ENTRADAS
% n_aval: n�mero de avalia��es da fun��o.
% problem: 1 para a dtlz1 ou 2 para a dtlz2.
% n_obj: 3 ou 5 objetivos.
% n_exec: n�mero de vezes que o algoritmo inteiro � executado.
% 
%SA�DAS
% ind_best: matriz contendo as vari�veis dos indiv�duos n�o dominados para o melhor valor de IGD obtido (IGD_best) em todas as n_exec execu��es.
% aval_best: matriz contendo a avalia��o dos indiv�duos n�o dominados para o melhor valor de IGD obtido (IGD_best) em todas as n_exec execu��es.
% IGD_best: melhor valor de IGD obtido nas n_exec execu��es.
% IGD_m: m�dia dos valores de IGD obtidos nas n_exec execu��es.
% ind_pior: matriz contendo as vari�veis dos indiv�duos n�o dominados para o pior valor de IGD obtido (IGD_pior) em todas as n_exec execu��es.
% aval_pior: matriz contendo a avalia��o dos indiv�duos n�o dominados para o pior valor de IGD obtido (IGD_pior) em todas as n_exec execu��es.
% IGD_pior: pior valor de IGD obtido nas n_exec execu��es.
    
    %configura��es iniciais
    clc;
    
    %configura��es do tipo de dist�ncia a ser utilizada no NSGAII
    tipo_dist = 'euclidean'; %tipo de dist�ncia para o c�lculo do IGD
    
    %configura��es do Diferencial Evolution Modificado
    CR = 0.9; %Taxa de cruzamento
    polaridadeTarget = 1; %Taxa que polariza o 'target' 
                          %em detrimento do vetor 'trial'
    %defini��o do intervalo de varia��o da taxa de muta��o (peso)
    Fmax = 1.2;
    Fmin = 0.2;
    
    %O valor de F decai ao longo das itera��es com base numa fun��o
    %log�stica de decaimento que tem curvatura baseada na taxa de
    %decaimento beta
    beta = calculaBeta(n_aval);
    
    qtdeIndividuos = 3; %Quantidade de individuos selecionados
    
    %vetor que armazena o resultado do c�lculo da m�trica de hipervolume 
    %IGD para as popula��es em geradas em todas as execu��es
    igdcompleto = zeros(1,n_exec);
    
    %seta par�metros conforme entradas
    %k � setado dentro das fun��es de benchmark, npop e nvar s�o 
    %pr�-determinados (conforme artigo)
    
    %Probabilidade de uso do algoritmo de clusteriza��o 
    %K Means nas amostras para sele��o de indiv�duos em
    %detrimento de sele��o aleat�ria (propKMeans)
    %O valor � setado em 0% em 5 objetivos por causa do
    %custo computacional que � alto.
                    
    %o n�mero de individuos e geracoes e determinado com os mesmos
    %valores do artigo: [DEB, Kalyanmoy e JAIN, Himanshu. �An Evolutionary Many-Objective Optimization
    %Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    %Problems With Box Contraints�. IEEE Transations on Evolutionary Computation, Vol. 18,
    %N�. 4, 2014.] - Este foi utilizado com benchmark

    if  problem == 1 && n_obj == 3
        npop = 91;
        nvar = 7;
        probKMeans = 1;
        benchmark = load('dtlz1_3d.mat');
    elseif problem == 1 && n_obj == 5
        npop = 210;
        nvar = 9;
        probKMeans = 0;
        benchmark = load('dtlz1_5d.mat');
    elseif problem == 2 && n_obj == 3
        npop = 91;
        nvar = 12;
        probKMeans = 1;
        benchmark = load('dtlz2_3d.mat');
    elseif problem == 2 && n_obj == 5
        npop = 210;
        nvar = 14;
        probKMeans = 0;
        benchmark = load('dtlz2_5d.mat');
    else
        msgbox('Indique 1 ou 2 para problema e 3 ou 5 para n�mero de objetivos!');
        error('Indique 1 ou 2 para problema e 3 ou 5 para n�mero de objetivos!');
    end

    %matriz que armazena as informa��es completas das solu��es trabalhadas
    %em todas as execu��es de todas as populacoes sendo:
    %'1 at� nvar': �ndividuos na popula��o com nvar dimens�es
    %'nvar+1 at� tamanho total da matriz': Avalia��es(fronteira) para n_obj
    tam = nvar+n_obj;
    popcompleta = repmat(struct('populacao', zeros(npop, tam)), n_exec, 1);
    
    %Ir� executar n_exec vezes todo o algoritmo e retornar� os melhores
    %resultados encontrados
    for i = 1 : n_exec
        fprintf(['Iniciando a execu��o:',num2str(i),' \n']);
        
        %Chamada ao Diferencial Evolution Modificado
        [ind_atuais, front_atuais] = diferencialEvolution(probKMeans, CR, polaridadeTarget, Fmax, Fmin, beta, n_aval, problem, npop, nvar, n_obj, qtdeIndividuos);
        
        %Insere a solu��o encontrada na execu��o i na matriz geral
        popcompleta(i).populacao(:,1:nvar) = ind_atuais;
        popcompleta(i).populacao(:,nvar+1:tam) = front_atuais;
        
        %Calcula IGD para a fronteira da popula��o corrente contra a 
        %fronteira problema do benchmark escolhido e armazena
        if  problem == 1 && n_obj == 3
            igdcompleto(i) = calculaIGD(front_atuais, benchmark.fronteiraReal, tipo_dist);
        elseif problem == 1 && n_obj == 5
            igdcompleto(i) = calculaIGD(front_atuais, benchmark.fronteiraReal, tipo_dist);
        elseif problem == 2 && n_obj == 3
            igdcompleto(i) = calculaIGD(front_atuais, benchmark.fronteiraReal, tipo_dist);
        elseif problem == 2 && n_obj == 5
            igdcompleto(i) = calculaIGD(front_atuais, benchmark.fronteiraReal, tipo_dist);
        end

    end

    %Melhor indiv�duo
    imelhorIndividuo = find(igdcompleto == min(igdcompleto));
    
    %Matriz contendo as vari�veis dos indiv�duos n�o dominados para o 
    %melhor valor de IGD obtido (IGD_best) em todas as n_exec execu��es.
    ind_best = popcompleta(imelhorIndividuo).populacao(:,1:nvar);
    
    %aval_best: matriz contendo a avalia��o dos indiv�duos n�o dominados 
    %para o melhor valor de IGD obtido (IGD_best) em todas as n_exec 
    %execu��es.
    aval_best = popcompleta(imelhorIndividuo).populacao(:,nvar+1:tam);
    
    %IGD_best: melhor valor de IGD obtido nas n_exec execu��es.
    IGD_best = igdcompleto(imelhorIndividuo);
    
    %Pior indiv�duo
    ipiorIndividuo = find(igdcompleto == max(igdcompleto));
    
    %ind_pior: matriz contendo as vari�veis dos indiv�duos n�o dominados 
    %para o pior valor de IGD obtido (IGD_pior) em todas as n_exec 
    %execu��es.
    ind_pior = popcompleta(ipiorIndividuo).populacao(:,1:nvar);
    
    %aval_pior: matriz contendo a avalia��o dos indiv�duos n�o dominados 
    %para o pior valor de IGD obtido (IGD_pior) em todas as n_exec 
    %execu��es.
    aval_pior = popcompleta(ipiorIndividuo).populacao(:,nvar+1:tam);
    
    %IGD_pior: pior valor de IGD obtido nas n_exec execu��es.
    IGD_pior = igdcompleto(ipiorIndividuo);
    
    %C�lculo do IGD-m
    %IGD_m: m�dia dos valores de IGD obtidos nas n_exec execu��es.
    IGD_m = mean(igdcompleto);

end

%Diferencial Evolution Modificado
function [individuos, fronteira] = diferencialEvolution(probKMeans, CR, poltarget, Fmax, Fmin, beta, n_aval, problem, npop, nvar, n_obj, qtdeIndividuos)
    %Gerando popula��o inicial
    [individuos, fronteira] = geraPopInicial(npop, nvar, problem, n_obj);
 
    %---------------------------------------------------------------------
    %Defini��o do menor e maior valor de F na curva da fun��o log�stica de
    %decaimento - valores usados para mapear F no intervalo determinado
    ngen = floor(n_aval/npop);
    Finf = (-(exp(ngen/beta)^2)*0.01+1);
    Fsup = (-(exp(1/beta)^2)*0.01+1);
    
    %Aplicando os operadores geneticos do Diferencial Evolution
    ncal = npop;
    gen = 1;
    while ncal < n_aval
        prog = (ncal*100)/n_aval;
        fprintf(['[',num2str(prog),'] \n']);
        %Aplica dist�ncia de multid�o entre os indiv�duos da popula��o
        %inicial para averiguar dominancias
        distMultidao = calculaDistMultidao(fronteira,npop,n_obj,1);
        
        %contador do custo de avalia��es feitas na popula�ao corrente
        icusto = 0;
        
        %Filhos a serem gerados pelo Diferencial Evolution
        filhos = zeros(npop,nvar);
        fronteira_filhos = zeros(npop,n_obj);  
        
        %Dinamicamente o valor de F � diminuido de gera��o em gera��o tendo
        %a explora��o X explota��o balanceada pelo valor de beta que
        %determina a curvatura da linha da fun��o log�stica de decaimento
        F = (-(exp(gen/beta)^2)*0.01+1);        
        F = ((F-Finf)/(Fsup-Finf))*(Fmax-Fmin)+Fmin;
            
        %Iniciando nova popula��o
        for i = 1:npop
            %seleciona-se os individuos que ir�o compor o vetor 'trial'
            selecionados = selecao(probKMeans, npop, fronteira, distMultidao, i, qtdeIndividuos);
            
            %Constr�i-se o vetor 'trial' com base nos par�metros de
            %muta��o
            trial = mutacao(individuos, selecionados, F);
            
            %define o individuo corrente como indiv�duo 'target'
            %individuo 'target'
            target =  individuos(i,:);
            %avaliacao do individuo 'target'
            fronttarget = fronteira(i,:);

            %Executa a opera��o de cruzamento entre o individuo corrente
            %'target' e o vetor de perturba��o 'trial' para gerar um novo
            %individuo na nova populacao
            filho = cruzamento(individuos, target, trial, CR, nvar, poltarget);
            
            %avalia o novo individuo gerado com base no benchmark escolhido            
            frontfilho = problema(problem, filho);

            %Avalia dominancia entre o filho gerado e o vetor 'target' e
            %insere o vencedor na nova popula��o
            [filhos(i,:), fronteira_filhos(i,:)] = avaliaIndividuo(target, filho, fronttarget,  frontfilho, fronteira, npop, n_obj, i, distMultidao);

            %Incrementa o contador de custo de avalia��o
            icusto = icusto + 1;
        end
        % ----------------------------------------------------------------
        
        %atualiza o contador de custo
        ncal = ncal + icusto;
        gen = gen + 1;
        %Popula��o antiga � substituida pela nova
        individuos = filhos;
        fronteira = fronteira_filhos;
    end
    
end

%Para o benchmark escolhido cria uma popula��o inicial aleatoriamente e
%na sequencia avalia a mesma gerando a avalia��o (fronteira) atual
function [individuos, fronteira] = geraPopInicial (npop, nvar, problem, n_obj)
    %gera npop individuos aleatoriamente
    fronteira = zeros(npop, n_obj);
    individuos = rand(npop, nvar);
    
    %avalia os individuos gerados segundo o benchmark escolhido
    for i=1:npop
        fronteira(i,:) = problema(problem,individuos(i,:));
    end
end

%Seleciona o problema a ser utilizado no c�lculo da fronteira
function [resp] = problema(problem, ind)
    if problem == 1
        resp = dtlz1(ind)';
    else
        resp = dtlz2(ind)';
    end
end

%Seleciona uma quantidade espec�fica de individuos aleatoriamente e mant�m,
%rodada a rodada, os que tem dominancia
function [selecionados] = selecao(probKMeans, npop, fronteira, distMultidao, iCorrente, qtdeIndividuos)
    %seleciona na popula��o aleatoriamente ou atrav�s do algoritmo de
    %clusteriza��o K Means com base em uma probabilidade os indiv�duos que
    %ir�o para a fase de cruzamento e muta��o
    if rand(1)<probKMeans
        candidatos = Kmeans(fronteira,qtdeIndividuos*2+1);
    else
        candidatos = randperm(npop,qtdeIndividuos*2+1);
    end
    
    %Verifica se algum individuo � igual ao individuoCorrente    
    nrepetidos = find(candidatos == iCorrente);
    
    %caso algum seja repetido ele � retirado da lista
    if nrepetidos ~= 0
        candidatos(nrepetidos(1)) = [];
    end
    
    %Avalia qual individuo domina o outro e seleciona o n�o dominado
    selecionados = zeros(1,qtdeIndividuos);
    j = 1;

    for i = 1:qtdeIndividuos
        %avalia a dominancia entre os dois primeiros individuos escolhidos 
        %aleatoriamente por meio das suas avalia��es
        dominado = calculaDominado(fronteira(candidatos(j),:),fronteira(candidatos(j+1),:));
        
        %caso n�o haja domin�ncia direta entre eles � aplicada ent�o a dist�ncia
        %de multid�o para escolher quem domina quem
        if dominado == 0
            %O candidato com mais vizinhos � dominado
            if distMultidao(candidatos(j)) < distMultidao(candidatos(j+1))
                dominado = 2;
            else
                dominado = 1;
            end
        end
        
        %Ap�s determina��o do dominado aloca-se o candidato escolhido a
        %lista dos selecionados
        if dominado == 1
            selecionados(i) = candidatos(j+1);
        else
            selecionados(i) = candidatos(j);
        end
        
        %Passa para a pr�xima dupla a ser avaliada
        j = j + 2;
    end
end

%Avalia por dominancia e dist�ncia de multid�o uma quantidade
%qtdeAvaliados selecionando apenas uma quantidade de qtdeIndividuos para
%comporem o vetor 'trial' que ir� para o cruzamento
function [trial] = mutacao(individuos, selecionados, F)
    %cria o vetor de perturba��o 'trial' sendo trial= xr3 + (F*(xr1-xr2))
    trial = individuos(selecionados(3),:) + F * (individuos(selecionados(1),:) - individuos(selecionados(2),:));
end

%Executa o cruzamento dos vetores 'target' e 'trial' levando-se em conta a
%taxa de cruzamento. Os vetores trocam posi��es caso a probabilidade
%ocorra.
function [novofilho] = cruzamento(indpop, target, trial, CR, nvar, poltarget)
    i = randi(nvar);
    probabilidade = 0;
     
    if rand()<poltarget
        %Probabilidade do vetor 'target' ser privilegiado no cruzamento
        novofilho = target;
        
        while  (probabilidade <= (1 - CR)) && (i <= nvar)
            if ( trial(i) < 0 || trial(i) > 1)               
                novofilho(i) = indpop(randi(nvar),i);
            else
                novofilho(i) = trial(i);
            end
            i = i + 1;

            probabilidade = rand();
        end    
    else
        %Probabilidade do vetor 'trial' ser privilegiado no cruzamento
        novofilho = trial;
        
        while  (probabilidade <= (1 - CR)) && (i <= nvar)
            if ( target(i) < 0 || target(i) > 1)               
                novofilho(i) = indpop(randi(nvar),i);
            else
                novofilho(i) = target(i);
            end
            i = i + 1;

            probabilidade = rand();
        end
    end
end

%Avalia o individuo 'target' e o filho gerado para saber quem vai para a
%nova popula��o
function [novoIndividuo, frontNovoIndividuo] = avaliaIndividuo(t, f, frontt, frontf, fronteira,npop,n_obj,iCorrente,distMult)
    dominado = calculaDominado(frontt,frontf);   
    
    %Caso nenhum domine nenhum
    if dominado == 0
        %calcula a dist�ncia de multid�o do filho gerado para a fronteira
        %e compara a mesma com a do indiv�duo 'target' (corrente)
        distMultidaoFilho = calculaDistMultidao(fronteira,npop,n_obj,2,frontf,iCorrente);
        if distMult(iCorrente) > distMultidaoFilho
           dominado = 1;
        else
           dominado = 2;
        end
    end
    
    %Retorna o dominante para a nova popula��o
    if dominado == 1
        novoIndividuo = t;
        frontNovoIndividuo = frontt;
    else
        novoIndividuo = f;
        frontNovoIndividuo = frontf;
    end
end

%Aplica as solu��es da fronteira o calculo da distancia de multidao
%conforme descrito no NSGAII
function [distMultidao] = calculaDistMultidao(fronteira, npop, nobj, tipo, frontfilho, iCorrente)
    %C�lculo da dist�ncia de multid�o entre a fronteira e a popula��o
    %corrente
    if tipo == 1
        distMultidao = zeros(1,npop);
        for i=1:nobj
            %ordena os valores das avalia�oes em cada objetivo
            [ordenado,indices] = sortrows(fronteira,i);

            %coleta a diferen�ao entre os valores m�ximos e m�nimos em cada objetivo
            difmaxmin = max(ordenado(:,i)) - min(ordenado(:,i));

            %aplica o limite infinito para a primeira solu��o
            distMultidao(indices(1)) = inf;
            %aplica o limite infinito para a �ltima solu��o
            distMultidao(indices(npop)) = inf;

            %calcula-se a dist�ncia de multid�o para todos os indiv�duos com
            %exce��o do primeiro e do �ltimo que s�o refer�ncias (setados com
            %infinito para suas proprias distancias)
            for j = 2:npop-1
                difpontos = ordenado(j+1,i) - ordenado(j-1,i);
                pontoatual = distMultidao(indices(j));
                distMultidao(indices(j)) = pontoatual + (difpontos/difmaxmin);
            end
        end
    else
    %C�lculo da dist�ncia de multid�o entre a fronteira e o filho gerado
    %pelo Diferencial Evolution
        fronteira(iCorrente,:) = frontfilho;
        distMultidao = 0;

        for i=1:nobj    
            %ordena os valores das avalia�oes em cada objetivo
            [ordenado,indices] = sortrows(fronteira,i);

            %coleta a diferen�ao entre os valores m�ximos e m�nimos em cada objetivo
            difmaxmin = max(ordenado(:,i)) - min(ordenado(:,i));
            
            %calcula-se a dist�ncia de multid�o da fronteira para o indice
            %da popula��o que substituiu com o filho gerado
            iOrdenado = find(indices == iCorrente);
            if iOrdenado ~= 1 && iOrdenado ~= npop
                difpontos = ordenado(iOrdenado+1,i) - ordenado(iOrdenado-1,i);
                distMultidao = distMultidao + (difpontos/difmaxmin);
            end
        end
    end
end

%Identifica se existe domina��o de um indiv�duo sobre o outro e qual � o
%dominado
function [dominado] = calculaDominado(individuo1, individuo2)
    %Nenhum domina nenhum
    dominado = 0;
    
    %Indiv�duo 1 � dominado por 2
    if all(individuo2 >= individuo1) && any(individuo1 < individuo2)
        dominado = 1;
    end
    
    %Indiv�duo 2 � dominado por 1
    if all(individuo1 >= individuo2) && any(individuo2 < individuo1)
        dominado = 2;
    end
end

%Calcula o IGD da fronteira atual versus a do benchmark escolhido
function [total] = calculaIGD(fatual, freal, tipo)
    [tfreal, ~] = size(freal);
    total = 0;

    %Toma a m�nima dist�ncia de todos os pontos da fronteira atual para 
    %cada ponto da fronteira do benchmark escolhido e extrai a m�dia
    for i = 1:tfreal
        [tfatual, ~] = size(fatual);
        disttodos = zeros(tfatual, 1);
        for j = 1:tfatual
            idist = pdist2(fatual(j,:),freal(i, :),tipo);
            disttodos(j, 1) = idist;
        end
        total = total + min(disttodos);
    end
    
    total = total / tfreal;
end

%Fun��o fornecida DTLZ1
function [F, varargout] = dtlz1(x, varargin)
    x = x(:);

    k = 5;
    n = length(x);
    m = n - k + 1;

    s = 0;
    for i = m:n
        s = s + (x(i)-0.5)^2 - cos(20*pi*(x(i)-0.5));
    end
    g = 100*(k+s);

    f(1) = 0.5 * prod(x(1:m-1)) * (1+g);
    for i = 2:m-1
        f(i) = 0.5 * prod(x(1:m-i)) * (1-x(m-i+1)) * (1+g);
    end
    f(m) = 0.5 * (1-x(1)) * (1+g);

    F = f(:);

    varargout(1) = {F};
    varargout(2) = {0};
end

%Fun��o fornecida DTLZ2
function [F, varargout] = dtlz2(x, varargin)

    x = x(:);

    k = 10;
    n = length(x);
    m = n - k + 1;

    s = 0;
    for i = m:n
    s = s + (x(i)-0.5)^2;
    end
    g = s;

    cosx = cos(x*pi/2);
    sinx = sin(x*pi/2);

    f(1) =  (1+g) * prod(cosx(1:m-1));
    for i = 2:m-1
    f(i) = (1+g) * prod(cosx(1:m-i)) * sinx(m-i+1);
    end
    f(m) = (1+g) * sinx(1);

    F = f(:);

    varargout(1) = {F};
    varargout(2) = {0};
end

%Calcula um numero ideal para a curvatura da fun��o logistica de decaimento
%que determina a taxa de decaimento de F ao longo das iteracoes
function beta = calculaBeta(ncal)
    if ncal <= 100
        beta = 2.5;
    elseif ncal > 100 && ncal <= 500
        beta = 8;
    elseif ncal > 500 && ncal <= 1000
        beta = 17;
    elseif ncal > 1000 && ncal <= 2000
        beta = 20;
    elseif ncal > 2000 && ncal <= 5000
        beta = 40;
    elseif ncal > 5000 && ncal <= 10000
        beta = 100;
    else
        beta = 170;
    end
end
