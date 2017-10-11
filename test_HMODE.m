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

%ENTRADAS
% n_aval: n�mero de avalia��es da fun��o.
% problem: 1 para a dtlz1 ou 2 para a dtlz2.
% n_obj: 3 ou 5 objetivos.
% n_exec: n�mero de vezes que o algoritmo inteiro � executado
%(para realizar diversos experimentos)

function [ind_best, aval_best] = test_HMODE(n_aval, problem, n_obj, n_exec)

    [ind_best, aval_best, IGD_best, IGD_m, ind_pior, aval_pior, IGD_pior] = Ciniro(n_aval, problem, n_obj, n_exec);
    
    fprintf('\t Melhor: %f \t Pior: %f \t M�dia: %f \n',IGD_best,IGD_pior,IGD_m);
    
    load('dtlz2_3d.mat');
    
    %grafico de plot para objetivos 3 e 5
%     figure();
%     hold on;
%     plot(aval_best')
%     hold off;
    
    %gr�fico 3d pra 3 objetivos
    figure();
    hold on;
    grid;
    scatter3(aval_best(:,1),aval_best(:,2),aval_best(:,3));
    
    if problem == 1
        x = [0.5 0.0 0.0 0.5];
        y = [0.0 0.5 0.0 0.0];
        z = [0.0 0.0 0.5 0.0];
        plot3(x, y, z);
    end
    


    %gr�fico 3d pra 3 objetivos
    scatter3(fronteiraReal(:,1),fronteiraReal(:,2),fronteiraReal(:,3));
    
    if problem == 1
        x = [0.5 0.0 0.0 0.5];
        y = [0.0 0.5 0.0 0.0];
        z = [0.0 0.0 0.5 0.0];
        plot3(x, y, z);
    end
    
    hold off;

end