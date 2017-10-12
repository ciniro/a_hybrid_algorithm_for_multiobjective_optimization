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

%                          ALGORITMO K MEANS                             %

%------------------------------------------------------------------------%

%x é a amostra de dados que será clusterizada
%qtde é a quantidade de centróides na inicializacao do k means

function [selecaofcm] = Kmeans(x,qtde)
    
    %agrupa os dados-------
    tam = size(x);
    nrows = tam(1);
    ncols = tam(2);
    ls = max(max(x));
    li = min(min(x));

    %Cálculo do tamanho de K
    k = 6;
    %coluna que vai armazenar a qual grupo pertence um dado padrão
    x = [x zeros(nrows,1)];

    tolerancia = 1e-5;
    parada = false;
    iteracoes = 0;

    %sorteia as médias
    mu = rand(k,ncols)*(ls - li)+li;

    %inicia a convergência
    while (parada==false)
        iteracoes = iteracoes + 1;
        mu_anterior = mu;

        %construção da matrix de dissimilaridade
        %Distância Euclideana
        dist = zeros(nrows,k);
        for cont=1:k
             euc = 0;
             for j=1:ncols
                 euc = euc + (x(:,j)-mu(cont,j)).^2;
             end
             dist(:,cont) = sqrt(euc);
        end

        %atribui grupo a um dado ponto
        x(:,(ncols+1))=0;
        [valor,grupo] = min(dist');
        x(:,(ncols+1))=grupo';

        %calcula os novos centroídes baseado na média dos grupos
        for cont=1:k
            for j=1:ncols 
                indices = find(x(:,(ncols+1))==cont);
                quantidade = size(indices);
                media = sum(x(indices,j))/quantidade(1);
                if media ~= 0 && isnan(media) ~= 1
                    mu(cont,j) = media;
                end
            end
        end

        %averigua a convergencia baseada em uma tolerancia mínima
        if sum(sum(abs(mu - mu_anterior)) > tolerancia)==0
            parada = true;
        end
    end
    
    %seleciona um de cada grupo
    grupos = randi([1 k],1,qtde);
    selecaofcm = zeros(1,qtde);
    cont = 1;
    while cont <= qtde
        elementosi = find(x(:,4) == grupos(cont));
        tam = size(elementosi);
        qtdeElementos = tam(1);
        if qtdeElementos ~= 0
            indiceElemento = randperm(qtdeElementos,1);
            selecaofcm(cont) = elementosi(indiceElemento);
            cont = cont + 1;
        else
            grupos(cont) = randi([1 k],1,1);
        end
    end
end

