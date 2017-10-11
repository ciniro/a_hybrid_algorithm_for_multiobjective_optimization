%------------------------------------------------------------------------%
%                  Universidade Federal de Minas Gerais                  %
%                  Departamento de Ci�ncia da Computa��o                 %
%                      Projeto - Computa��o Natural                      %
%                          Prof. Gisele L. Pappa                         %
%                                                                        %
%                 Aluno: CINIRO APARECIDO LEITE NAMETALA                 %
%------------------------------------------------------------------------%

%                          ALGORITMO K MEANS                             %

%------------------------------------------------------------------------%

%x � a amostra de dados que ser� clusterizada
%qtde � a quantidade de centr�ides na inicializacao do k means

function [selecaofcm] = Kmeans(x,qtde)
    
    %agrupa os dados-------
    tam = size(x);
    nrows = tam(1);
    ncols = tam(2);
    ls = max(max(x));
    li = min(min(x));

    %C�lculo do tamanho de K
    k = 6;
    %coluna que vai armazenar a qual grupo pertence um dado padr�o
    x = [x zeros(nrows,1)];

    tolerancia = 1e-5;
    parada = false;
    iteracoes = 0;

    %sorteia as m�dias
    mu = rand(k,ncols)*(ls - li)+li;

    %inicia a converg�ncia
    while (parada==false)
        iteracoes = iteracoes + 1;
        mu_anterior = mu;

        %constru��o da matrix de dissimilaridade
        %Dist�ncia Euclideana
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

        %calcula os novos centro�des baseado na m�dia dos grupos
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

        %averigua a convergencia baseada em uma tolerancia m�nima
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

