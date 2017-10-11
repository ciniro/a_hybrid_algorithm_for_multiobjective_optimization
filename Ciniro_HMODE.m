%------------------------------------------------------------------------%
%          XIII Brazilian Congress on Computational Intelligence         %
%                      Niterói, Rio de Janeiro, Brazil                   %
%                         2017-30-10 to 2017-11-01                       %
%                                                                        %
%                      CINIRO APARECIDO LEITE NAMETALA                   %
%                         (IFMG, Bambuí, MG, Brazil)                     %
%                                                                        %
%                              GISELE L. PAPPA                           %
%                          EDUARDO GONTIJO CARRANO                       %
%                     (UFMG, Belo Horizonte, MG, Brazil)                 %
%------------------------------------------------------------------------%
%                                                                        %
%            EVOLUÇÃO DIFERENCIAL HÍBRIDO COM K-MEANS E NSGA II          %
%            A Hybrid Algorithm for Multiobjective Optimization          %
%                        (MODE + K means + NSGA II)                      %
%                                                                        %
%------------------------------------------------------------------------%

function [ind_best, aval_best, IGD_best, IGD_m, ind_pior, aval_pior, IGD_pior] = Ciniro_HMODE(n_aval, problem, n_obj, n_exec)
%
%ENTRADAS
% n_aval: número de avaliações da função.
% problem: 1 para a dtlz1 ou 2 para a dtlz2.
% n_obj: 3 ou 5 objetivos.
% n_exec: número de vezes que o algoritmo inteiro é executado.
% 
%SAÍDAS
% ind_best: matriz contendo as variáveis dos indivíduos não dominados para o melhor valor de IGD obtido (IGD_best) em todas as n_exec execuções.
% aval_best: matriz contendo a avaliação dos indivíduos não dominados para o melhor valor de IGD obtido (IGD_best) em todas as n_exec execuções.
% IGD_best: melhor valor de IGD obtido nas n_exec execuções.
% IGD_m: média dos valores de IGD obtidos nas n_exec execuções.
% ind_pior: matriz contendo as variáveis dos indivíduos não dominados para o pior valor de IGD obtido (IGD_pior) em todas as n_exec execuções.
% aval_pior: matriz contendo a avaliação dos indivíduos não dominados para o pior valor de IGD obtido (IGD_pior) em todas as n_exec execuções.
% IGD_pior: pior valor de IGD obtido nas n_exec execuções.
    
    %configurações iniciais
    clc;
    
    %configurações do tipo de distância a ser utilizada no NSGAII
    tipo_dist = 'euclidean'; %tipo de distância para o cálculo do IGD
    
    %configurações do Diferencial Evolution Modificado
    CR = 0.9; %Taxa de cruzamento
    polaridadeTarget = 1; %Taxa que polariza o 'target' 
                          %em detrimento do vetor 'trial'
    %definição do intervalo de variação da taxa de mutação (peso)
    Fmax = 1.2;
    Fmin = 0.2;
    
    %O valor de F decai ao longo das iterações com base numa função
    %logística de decaimento que tem curvatura baseada na taxa de
    %decaimento beta
    beta = calculaBeta(n_aval);
    
    qtdeIndividuos = 3; %Quantidade de individuos selecionados
    
    %vetor que armazena o resultado do cálculo da métrica de hipervolume 
    %IGD para as populações em geradas em todas as execuções
    igdcompleto = zeros(1,n_exec);
    
    %seta parâmetros conforme entradas
    %k é setado dentro das funções de benchmark, npop e nvar são 
    %pré-determinados (conforme artigo)
    
    %Probabilidade de uso do algoritmo de clusterização 
    %K Means nas amostras para seleção de indivíduos em
    %detrimento de seleção aleatória (propKMeans)
    %O valor esta setado como padrão em 0% em 5 objetivos por causa do
    %custo computacional que é alto. Modifique isso para realizar seus testes, se desejar.
                    
    %o número de individuos e geracoes e determinado com os mesmos
    %valores do artigo: [DEB, Kalyanmoy e JAIN, Himanshu. “An Evolutionary Many-Objective Optimization
    %Algorithm Using Reference-Point-Based Nondominated Sorting Approach, Part I: Solving
    %Problems With Box Contraints”. IEEE Transations on Evolutionary Computation, Vol. 18,
    %Nº. 4, 2014.] - Este foi utilizado com benchmark

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
        msgbox('Indique 1 ou 2 para problema e 3 ou 5 para número de objetivos!');
        error('Indique 1 ou 2 para problema e 3 ou 5 para número de objetivos!');
    end

    %matriz que armazena as informações completas das soluções trabalhadas
    %em todas as execuções de todas as populacoes sendo:
    %'1 até nvar': Índividuos na população com nvar dimensões
    %'nvar+1 até tamanho total da matriz': Avaliações(fronteira) para n_obj
    tam = nvar+n_obj;
    popcompleta = repmat(struct('populacao', zeros(npop, tam)), n_exec, 1);
    
    %Irá executar n_exec vezes todo o algoritmo e retornará os melhores
    %resultados encontrados
    for i = 1 : n_exec
        fprintf(['Iniciando a execução:',num2str(i),' \n']);
        
        %Chamada ao Diferencial Evolution Modificado
        [ind_atuais, front_atuais] = diferencialEvolution(probKMeans, CR, polaridadeTarget, Fmax, Fmin, beta, n_aval, problem, npop, nvar, n_obj, qtdeIndividuos);
        
        %Insere a solução encontrada na execução i na matriz geral
        popcompleta(i).populacao(:,1:nvar) = ind_atuais;
        popcompleta(i).populacao(:,nvar+1:tam) = front_atuais;
        
        %Calcula IGD para a fronteira da população corrente contra a 
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

    %Melhor indivíduo
    imelhorIndividuo = find(igdcompleto == min(igdcompleto));
    
    %Matriz contendo as variáveis dos indivíduos não dominados para o 
    %melhor valor de IGD obtido (IGD_best) em todas as n_exec execuções.
    ind_best = popcompleta(imelhorIndividuo).populacao(:,1:nvar);
    
    %aval_best: matriz contendo a avaliação dos indivíduos não dominados 
    %para o melhor valor de IGD obtido (IGD_best) em todas as n_exec 
    %execuções.
    aval_best = popcompleta(imelhorIndividuo).populacao(:,nvar+1:tam);
    
    %IGD_best: melhor valor de IGD obtido nas n_exec execuções.
    IGD_best = igdcompleto(imelhorIndividuo);
    
    %Pior indivíduo
    ipiorIndividuo = find(igdcompleto == max(igdcompleto));
    
    %ind_pior: matriz contendo as variáveis dos indivíduos não dominados 
    %para o pior valor de IGD obtido (IGD_pior) em todas as n_exec 
    %execuções.
    ind_pior = popcompleta(ipiorIndividuo).populacao(:,1:nvar);
    
    %aval_pior: matriz contendo a avaliação dos indivíduos não dominados 
    %para o pior valor de IGD obtido (IGD_pior) em todas as n_exec 
    %execuções.
    aval_pior = popcompleta(ipiorIndividuo).populacao(:,nvar+1:tam);
    
    %IGD_pior: pior valor de IGD obtido nas n_exec execuções.
    IGD_pior = igdcompleto(ipiorIndividuo);
    
    %Cálculo do IGD-m
    %IGD_m: média dos valores de IGD obtidos nas n_exec execuções.
    IGD_m = mean(igdcompleto);

end

%Diferencial Evolution Modificado
function [individuos, fronteira] = diferencialEvolution(probKMeans, CR, poltarget, Fmax, Fmin, beta, n_aval, problem, npop, nvar, n_obj, qtdeIndividuos)
    %Gerando população inicial
    [individuos, fronteira] = geraPopInicial(npop, nvar, problem, n_obj);
 
    %---------------------------------------------------------------------
    %Definição do menor e maior valor de F na curva da função logística de
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
        %Aplica distância de multidão entre os indivíduos da população
        %inicial para averiguar dominancias
        distMultidao = calculaDistMultidao(fronteira,npop,n_obj,1);
        
        %contador do custo de avaliações feitas na populaçao corrente
        icusto = 0;
        
        %Filhos a serem gerados pelo Diferencial Evolution
        filhos = zeros(npop,nvar);
        fronteira_filhos = zeros(npop,n_obj);  
        
        %Dinamicamente o valor de F é diminuido de geração em geração tendo
        %a exploração X explotação balanceada pelo valor de beta que
        %determina a curvatura da linha da função logística de decaimento
        F = (-(exp(gen/beta)^2)*0.01+1);        
        F = ((F-Finf)/(Fsup-Finf))*(Fmax-Fmin)+Fmin;
            
        %Iniciando nova população
        for i = 1:npop
            %seleciona-se os individuos que irão compor o vetor 'trial'
            selecionados = selecao(probKMeans, npop, fronteira, distMultidao, i, qtdeIndividuos);
            
            %Constrói-se o vetor 'trial' com base nos parâmetros de
            %mutação
            trial = mutacao(individuos, selecionados, F);
            
            %define o individuo corrente como indivíduo 'target'
            %individuo 'target'
            target =  individuos(i,:);
            %avaliacao do individuo 'target'
            fronttarget = fronteira(i,:);

            %Executa a operação de cruzamento entre o individuo corrente
            %'target' e o vetor de perturbação 'trial' para gerar um novo
            %individuo na nova populacao
            filho = cruzamento(individuos, target, trial, CR, nvar, poltarget);
            
            %avalia o novo individuo gerado com base no benchmark escolhido            
            frontfilho = problema(problem, filho);

            %Avalia dominancia entre o filho gerado e o vetor 'target' e
            %insere o vencedor na nova população
            [filhos(i,:), fronteira_filhos(i,:)] = avaliaIndividuo(target, filho, fronttarget,  frontfilho, fronteira, npop, n_obj, i, distMultidao);

            %Incrementa o contador de custo de avaliação
            icusto = icusto + 1;
        end
        % ----------------------------------------------------------------
        
        %atualiza o contador de custo
        ncal = ncal + icusto;
        gen = gen + 1;
        %População antiga é substituida pela nova
        individuos = filhos;
        fronteira = fronteira_filhos;
    end
    
end

%Para o benchmark escolhido cria uma população inicial aleatoriamente e
%na sequencia avalia a mesma gerando a avaliação (fronteira) atual
function [individuos, fronteira] = geraPopInicial (npop, nvar, problem, n_obj)
    %gera npop individuos aleatoriamente
    fronteira = zeros(npop, n_obj);
    individuos = rand(npop, nvar);
    
    %avalia os individuos gerados segundo o benchmark escolhido
    for i=1:npop
        fronteira(i,:) = problema(problem,individuos(i,:));
    end
end

%Seleciona o problema a ser utilizado no cálculo da fronteira
function [resp] = problema(problem, ind)
    if problem == 1
        resp = dtlz1(ind)';
    else
        resp = dtlz2(ind)';
    end
end

%Seleciona uma quantidade específica de individuos aleatoriamente e mantém,
%rodada a rodada, os que tem dominancia
function [selecionados] = selecao(probKMeans, npop, fronteira, distMultidao, iCorrente, qtdeIndividuos)
    %seleciona na população aleatoriamente ou através do algoritmo de
    %clusterização K Means com base em uma probabilidade os indivíduos que
    %irão para a fase de cruzamento e mutação
    if rand(1)<probKMeans
        candidatos = Kmeans(fronteira,qtdeIndividuos*2+1);
    else
        candidatos = randperm(npop,qtdeIndividuos*2+1);
    end
    
    %Verifica se algum individuo é igual ao individuoCorrente    
    nrepetidos = find(candidatos == iCorrente);
    
    %caso algum seja repetido ele é retirado da lista
    if nrepetidos ~= 0
        candidatos(nrepetidos(1)) = [];
    end
    
    %Avalia qual individuo domina o outro e seleciona o não dominado
    selecionados = zeros(1,qtdeIndividuos);
    j = 1;

    for i = 1:qtdeIndividuos
        %avalia a dominancia entre os dois primeiros individuos escolhidos 
        %aleatoriamente por meio das suas avaliações
        dominado = calculaDominado(fronteira(candidatos(j),:),fronteira(candidatos(j+1),:));
        
        %caso não haja dominância direta entre eles é aplicada então a distância
        %de multidão para escolher quem domina quem
        if dominado == 0
            %O candidato com mais vizinhos é dominado
            if distMultidao(candidatos(j)) < distMultidao(candidatos(j+1))
                dominado = 2;
            else
                dominado = 1;
            end
        end
        
        %Após determinação do dominado aloca-se o candidato escolhido a
        %lista dos selecionados
        if dominado == 1
            selecionados(i) = candidatos(j+1);
        else
            selecionados(i) = candidatos(j);
        end
        
        %Passa para a próxima dupla a ser avaliada
        j = j + 2;
    end
end

%Avalia por dominancia e distância de multidão uma quantidade
%qtdeAvaliados selecionando apenas uma quantidade de qtdeIndividuos para
%comporem o vetor 'trial' que irá para o cruzamento
function [trial] = mutacao(individuos, selecionados, F)
    %cria o vetor de perturbação 'trial' sendo trial= xr3 + (F*(xr1-xr2))
    trial = individuos(selecionados(3),:) + F * (individuos(selecionados(1),:) - individuos(selecionados(2),:));
end

%Executa o cruzamento dos vetores 'target' e 'trial' levando-se em conta a
%taxa de cruzamento. Os vetores trocam posições caso a probabilidade
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
%nova população
function [novoIndividuo, frontNovoIndividuo] = avaliaIndividuo(t, f, frontt, frontf, fronteira,npop,n_obj,iCorrente,distMult)
    dominado = calculaDominado(frontt,frontf);   
    
    %Caso nenhum domine nenhum
    if dominado == 0
        %calcula a distância de multidão do filho gerado para a fronteira
        %e compara a mesma com a do indivíduo 'target' (corrente)
        distMultidaoFilho = calculaDistMultidao(fronteira,npop,n_obj,2,frontf,iCorrente);
        if distMult(iCorrente) > distMultidaoFilho
           dominado = 1;
        else
           dominado = 2;
        end
    end
    
    %Retorna o dominante para a nova população
    if dominado == 1
        novoIndividuo = t;
        frontNovoIndividuo = frontt;
    else
        novoIndividuo = f;
        frontNovoIndividuo = frontf;
    end
end

%Aplica as soluções da fronteira o calculo da distancia de multidao
%conforme descrito no NSGAII
function [distMultidao] = calculaDistMultidao(fronteira, npop, nobj, tipo, frontfilho, iCorrente)
    %Cálculo da distância de multidão entre a fronteira e a população
    %corrente
    if tipo == 1
        distMultidao = zeros(1,npop);
        for i=1:nobj
            %ordena os valores das avaliaçoes em cada objetivo
            [ordenado,indices] = sortrows(fronteira,i);

            %coleta a diferençao entre os valores máximos e mínimos em cada objetivo
            difmaxmin = max(ordenado(:,i)) - min(ordenado(:,i));

            %aplica o limite infinito para a primeira solução
            distMultidao(indices(1)) = inf;
            %aplica o limite infinito para a última solução
            distMultidao(indices(npop)) = inf;

            %calcula-se a distância de multidão para todos os indivíduos com
            %exceção do primeiro e do último que são referências (setados com
            %infinito para suas proprias distancias)
            for j = 2:npop-1
                difpontos = ordenado(j+1,i) - ordenado(j-1,i);
                pontoatual = distMultidao(indices(j));
                distMultidao(indices(j)) = pontoatual + (difpontos/difmaxmin);
            end
        end
    else
    %Cálculo da distância de multidão entre a fronteira e o filho gerado
    %pelo Diferencial Evolution
        fronteira(iCorrente,:) = frontfilho;
        distMultidao = 0;

        for i=1:nobj    
            %ordena os valores das avaliaçoes em cada objetivo
            [ordenado,indices] = sortrows(fronteira,i);

            %coleta a diferençao entre os valores máximos e mínimos em cada objetivo
            difmaxmin = max(ordenado(:,i)) - min(ordenado(:,i));
            
            %calcula-se a distância de multidão da fronteira para o indice
            %da população que substituiu com o filho gerado
            iOrdenado = find(indices == iCorrente);
            if iOrdenado ~= 1 && iOrdenado ~= npop
                difpontos = ordenado(iOrdenado+1,i) - ordenado(iOrdenado-1,i);
                distMultidao = distMultidao + (difpontos/difmaxmin);
            end
        end
    end
end

%Identifica se existe dominação de um indivíduo sobre o outro e qual é o
%dominado
function [dominado] = calculaDominado(individuo1, individuo2)
    %Nenhum domina nenhum
    dominado = 0;
    
    %Indivíduo 1 é dominado por 2
    if all(individuo2 >= individuo1) && any(individuo1 < individuo2)
        dominado = 1;
    end
    
    %Indivíduo 2 é dominado por 1
    if all(individuo1 >= individuo2) && any(individuo2 < individuo1)
        dominado = 2;
    end
end

%Calcula o IGD da fronteira atual versus a do benchmark escolhido
function [total] = calculaIGD(fatual, freal, tipo)
    [tfreal, ~] = size(freal);
    total = 0;

    %Toma a mínima distância de todos os pontos da fronteira atual para 
    %cada ponto da fronteira do benchmark escolhido e extrai a média
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

%Função fornecida DTLZ1
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

%Função fornecida DTLZ2
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

%Calcula um numero ideal para a curvatura da função logistica de decaimento
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
