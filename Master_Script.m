% % Start-up for dummies
% cd('/Applications/MATLAB/Neo.Matlib') Don't need to do anymore.. added to
% path :) 
run('/Applications/MATLAB/Neo.Matlib/add_libs_rnt.m')
run('/Applications/MATLAB/Neo.Matlib/startup.m')

cd('/Users/owner042/Dropbox/Acid-Namrata')
file = 'ph_Oyr_GFDL-ESM2G_historical_r1i1p1_1861-2005.nc';

%% mean pH in CCS over depths

nc = netcdf(file);      % makes 'nc' the file handle
upper_lev_range = find(nc{'lev'}>=100 & nc{'lev'}<=400);
surface_lev = upper_lev_range(1);
bottom_lev = upper_lev_range(end); % will use later

ph = nc{'ph'}(145,surface_lev,120:170,100:180); % ie. 2005, 100m depth, all lats+lons
ph (ph > 1.0e+15) = nan;
ph = perm(ph);
clf; 
pcolorjw(ph'); colorbar; 
caxis([7.4 8.35]);
title('Ocean pH in 2005, Depth: 100m')

ph = nc{'ph'}(145,bottom_lev,120:170,100:180); % ie. 2005, 100m depth, all lats+lons
ph (ph > 1.0e+15) = nan;
ph = perm(ph);
figure; clf; 
pcolorjw(ph'); colorbar; 
caxis([7.4 8.35]);
title('Ocean pH in 2005, Depth: 400m')


%% mean pH in the Pacific over time

year1 = 1861;
for count = 1:145
    ph = nc{'ph'}(count,surface_lev:bottom_lev,:,:); % ie.1861 (first loop), 100-400m, lats+lons
    ph (ph > 1.0e+15) = nan;
    ph = nanmean(ph,1);
    ph = perm(ph); clf; pcolorjw(ph'); colorbar; caxis([7.5 8.3])
    axis([25 210 20 180]) %zooms in on Pacific ocean, top to bottom
    title(sprintf('Mean Pacific pH in %d (100-400m)',year1));
    year1 = year1+1;
    pause(0.01)
    N(count) = getframe(gcf);
end
movie2avi(N, 'Pacific_mean_1861_2005.avi')

%% mean pH in CCS over time

year1 = 1861;
for count = 1:145
    ph = nc{'ph'}(count,surface_lev:bottom_lev,:,:); % ie.1861 (first loop), 100-400m, lats+lons
    ph (ph > 1.0e+15) = nan;
    ph = nanmean(ph,1);
    ph = perm(ph); clf; pcolorjw(ph'); colorbar; caxis([7.45 8.3])
    axis([120 170 100 180]) %zooms in on California coast
    title(sprintf('Mean California pH in %d (100-400m)',year1));
    year1 = year1+1;
    pause(0.01)
    N(count) = getframe(gcf);
end
movie2avi(N, 'California_mean_1861_2005.avi')


%% creating new data structure from 1861-2100 where mean is based on 1861-2005 

cd('/Users/owner042/Dropbox/2015-Spring/Data-Analysis/Final')
load('iso_data_1861_2005_2.mat')
earl = data_pre;

load('iso_data_ALL_1861_2100_2.mat')
data = saved_alldata;

data2.iso = data.iso;
data2.lon = data.lon;
data2.lat = data.lat;
data2.z_iso = data.z_iso;
mask = mean(data.z_iso(:,:,:),3); 
mask(~isnan(mask))=1;
data2.mask = data.mask;
data2.year = 1861:2100;
data2.z_iso = data.z_iso;
data2.t_iso = data.t_iso;
data2.s_iso = data.s_iso;
data2.ph_iso = data.ph_iso;
data2.do_iso = data.do_iso;
Tend = length(data2.year);

fields = {'z_iso' 's_iso' 't_iso' 'ph_iso' 'do_iso'};
fieldsa = {'za_iso' 'sa_iso' 'ta_iso' 'pha_iso' 'doa_iso'};
for i = 1 : 5
    field = getfield(earl, fields{i}); % this field has til 2005 data
    field2 = getfield(data2,fields{i}); % this field has til 2100 data
    fielda = field2 - repmat(mean(field,3), [1 1 Tend]);
    str=['data2.',fieldsa{i},' = fielda;']; eval(str);
end
mdtm_data = data2;
save mdtm_data_1861_2100.mat mdtm_data










%% START HERE 

%% Time series - whole time period, other stats too

cd('/Users/owner042/Dropbox/2015-Spring/Data-Analysis/Final')
load ('mdtm_data_1861_2100.mat')
data = mdtm_data;

xlim = [-136 -120]; ylim = [34 45];
[i,j]=rgrd_FindIJ(data.lon, data.lat, xlim, ylim);

s_index = sq(meanNaN(meanNaN(data.sa_iso(i,j,:),1),2));
t_index = sq(meanNaN(meanNaN(data.ta_iso(i,j,:),1),2));
ph_index = sq(meanNaN(meanNaN(data.pha_iso(i,j,:),1),2));
do_index = sq(meanNaN(meanNaN(data.doa_iso(i,j,:),1),2));

figure;
plot(data.year, nn(s_index)); hold on % nn = Neural Network Toolbox Utility Functions
plot(data.year, nn(t_index), 'r');
plot(data.year, nn(ph_index), 'm');
plot(data.year, nn(do_index), 'g'); % what do these values mean exactly?
xlabel('Year'); ylabel('Anomalies from 1861-2005 mean');
legend('salinity', 'temperature', 'pH', 'DO')
hold off









%% pH,natural portion

yvals1 = nn(ph_index);
yvals1 = yvals1(1:120);
xvals1 = data.year;
xvals1 = xvals1(1:120);
figure; subplot(2,1,1)
plot(xvals1, yvals1, 'm'); % real data
xlabel('Year'); ylabel('pH anomaly');
legend('Unmanipulated data')

subplot(2,1,2); 
plot(xvals1, yvals1, 'm'); hold on % real data
yvals11 = detrend(yvals1);
plot(xvals1,yvals11, 'k') % detrended data 
xlabel('Year'); ylabel('pH anomaly'); 
legend('Unmanipulated data', 'Detrended data')
hold off

    % checking for autocorrelation 
[cov_y, lags]=xcov(yvals11,100,'unbiased');
figure; stem(lags,cov_y)
    % none found (very similar to white noise graph)

    % me generating a signal to fit the data
prd = 20; % period (yrs)
b = 2.*pi./prd;
a = 0.15; %amplitude
d = 0; %vertical shift
hrz = 40; %horizontal shift
c = b.*hrz;

t1 = xvals1;
model1 = d+a*sin(b.*(t1-c));

figure; subplot(2,1,1);
plot(xvals1,yvals11, 'k') % detrended data
hold on    
plot(t1,model1,'g');
legend('Detrended data','Self-generated model')
xlabel('Year'); ylabel('pH anomaly')
hold off

    % strength of my own model
subplot(2,1,2);
plot(yvals11,model1,'bo')
xlabel('pH anomaly from data')
ylabel('pH anomaly from created model')
hold on
p = polyfit(yvals11',model1,1);
linreg = polyval(p,yvals11');
plot(yvals11, linreg, 'k--');
header = sprintf('Slope is %d',p(1));
text(0.1,-0.17,header)
title('Strength of created model')
hold off

    % using nlinfit to generate a model to fit the data
model = @(bel,t1) ((bel(1).*sin(bel(2).*(t1-bel(3))))+(bel(4).*cos(bel(5).*(t1-bel(6))))-(bel(7).*cos(bel(8).*(t1-bel(9))))+(bel(10).*cos(bel(11).*(t1-bel(12)))));
beta0 = [0.15;0.3142;12.56;0.15;0.3142;12.56;0.15;0.3142;12.56;0.15;0.3142;12.56];
beta=nlinfit(t1,yvals11',model,beta0);
modelys = (beta(1).*sin(beta(2).*(t1-beta(3)))) + (beta(4).*cos(beta(5).*(t1-beta(6))))-(beta(7).*cos(beta(8).*(t1-beta(9))))+(beta(10).*cos(beta(11).*(t1-beta(12))));

figure; subplot(2,1,1);
plot(t1,yvals11,'b'); hold on
%plot(t1,model1,'g');
plot(t1,modelys,'k'); legend('detrended', 'my model','nlinfit model')
hold off

    % checking strength of nlinfit model
subplot(2,1,2)
plot(yvals11,modelys,'bo')
xlabel('pH anomaly from data')
ylabel('pH anomaly from generated model')
hold on
p2 = polyfit(yvals11',modelys,1);
linreg2 = polyval(p2,yvals11');
plot(yvals11, linreg2, 'k--');
header = sprintf('Slope is %d',p2(1));
text(0.1,-0.1,header)
title('Strength of generated model')
hold off

% % Periodogram - means nothing b/c it's not a graph of occurrences, pH has a
% % range of values.. 
%     figure;
%     [Pxx,f]=periodogram(yvals11,[],length(t1),1./20);
%     plot(f,abs(Pxx));
%     xlabel('Frequency')
%     ylabel('Power')
%     % peak found at 0.0004 and 0.0021 yrs 

    % periodogram
nfft = length(yvals11);
fs = 200;
[Pxx,f] = periodogram(yvals11,[],nfft,fs);
semilogx(1./f(2:end),Pxx(2:end))
xlabel('Period'); ylabel('PSD')
title('Periodogram of Regime 1, detrended data')

    % lombscargle
xvals11 = xvals1';
matr = [xvals1' yvals11];
[xp xf] = lombscargle2(matr,2,4); % given 2 and 4; also don't have to run multiple times after the first time
figure; plot(1./xf,xp,'k'); 
axis([2 55 0 28])
xlabel('Period (years)')
ylabel('Power')
title('Least-squares spectral analysis of Regime 1')

    % first order fit

figure; subplot(2,1,1)
plot(xvals1, yvals1,'mo'); hold on

xlabel('Year'); ylabel('pH anomaly');
[j1 e1] = polyfit(xvals1,yvals1',1);
[fitvals9 d1] = polyval(j1,xvals1,e1);
title('linear fit for Regime 1')
plot(xvals1,fitvals9,'b'); 
plot(xvals1,fitvals9-d1, 'r--');
plot(xvals1,fitvals9+d1, 'r--');
hold off
legend('detrended pH', 'third-order fit')

subplot(2,1,2)
lin_res9 = yvals1'-fitvals9;
stem(xvals1, lin_res9, 'fill','--', 'MarkerFaceColor','r')
title('linear regression residuals')
ylabel('pH anomaly')

[r7 p7 rlow7 rhigh7] = corrcoef(yvals1',fitvals9)

some = polyfit(yvals1',fitvals9,1); % slope = 0.5968
stuff = polyval(some,yvals1');
figure;
plot(yvals1',fitvals9,'ro',yvals1',stuff,'k-')
    % slope error
errvar = sum(((fitvals9-stuff).^2)/(length(yvals1')-2));
slopestd = sqrt(errvar./var(yvals1'));
tval = abs(tinv(0.025,length(yvals1')-2));
low = some(1)-(tval.*slopestd)
high = some(1)+(tval.*slopestd)
some(1)
ylabel('fitted (model) values')
xlabel('data values')
title('Slope is %d',some(1))
legend('Predicted values', 'Linear fit')

% Linear regression is valid when error terms (residuals)
% 1) are normally distributed around 0
% 2) have a constant variance
% 3) are independent from each other

    % 1)
figure
% bins = round(sqrt(length(lin_res9)));
bins = 20
[lin9 centers] = hist(lin_res9,bins);
hist(lin_res9,bins)
hold on
xlabel('Residual')
ylabel('Occurrences')
title('Distribution of residuals for linear fit to Regime 1')
created9 = pdf('norm',centers,mean(lin_res9),std(lin_res9)); 
created_new9 = created9.*(sum(lin9)./sum(created9));
lin_num9 = (lin9 - created_new9).^2;
chi2_lin9 = sum(lin_num9./created_new9) % 23.95
chi2_lin_mark9 = chi2inv(0.95,bins-2-1) % 15.51
    % Chi2_lin_mark < chi2_lin_mark, so hypothesis cannot be rejected 
    % ie. Yes, probably normal!

hold off









%% pH, anthro portion

yvals2 = nn(ph_index);
yvals2 = yvals2(121:240);
xvals2 = data.year;
xvals2 = xvals2(121:240);

% anthro portion

    % autocorrelation

figure; subplot(2,1,1)
plot(xvals2, yvals2, 'm'); % real data
xlabel('Year'); ylabel('pH anomaly');
legend('Unmanipulated data')

subplot(2,1,2); 
plot(xvals2, yvals2, 'm'); hold on % real data
yvals22 = detrend(yvals2);
plot(xvals2,yvals22, 'k') % detrended data 
xlabel('Year'); ylabel('pH anomaly'); 
legend('Unmanipulated data', 'Detrended data')
hold off

[cov_y2, lags]=xcov(yvals22,100,'unbiased');
figure; stem(lags,cov_y2)

    % LSSA
xvals22 = xvals2';
matr2 = [xvals2' yvals22];
[xp2 xf2] = lombscargle2(matr2,2,4); % given 2 and 4; also don't have to run multiple times after the first time
figure; plot(1./xf2(2:end),xp2(2:end),'k'); 
axis([2 59 0 28])
xlabel('Period (years)')
ylabel('Power')
title('Least-squares spectral analysis of Regime 2')


    % first order 
figure; subplot(2,1,1)
plot(xvals2, yvals2,'mo'); hold on

xlabel('Year'); ylabel('pH anomaly');
[p4 e4] = polyfit(xvals2,yvals2',1);
[fitvals4 d2] = polyval(p4,xvals2,e4);
plot(xvals2,fitvals4,'b');
plot(xvals2,fitvals4-d2, 'r--')
plot(xvals2,fitvals4+d2, 'r--')
hold off
title('linear fit for Regime 2')
legend('detrended pH', 'first-order fit')
hold off

subplot(2,1,2)
lin_res4 = yvals2'-fitvals4;
stem(xvals2, lin_res4, 'fill','--', 'MarkerFaceColor','r'); 
title('first-order linear regression residuals')
ylabel('pH anomaly')
xlabel('Year')

    % normality of residuals
figure
bins = 20;
%bins = round(sqrt(length(lin_res4)));
[lin1 centers] = hist(lin_res4,bins);
hist(lin_res4,bins)
xlabel('Residual')
ylabel('Occurrences')
title('Distribution of residuals for first-order fit')
created1 = pdf('norm',centers,mean(lin_res4),std(lin_res4)); 
created_new = created1.*(sum(lin1)./sum(created1));
lin_num = (lin1 - created_new).^2;
chi2_lin = sum(lin_num./created_new) 
chi2_lin_mark = chi2inv(0.95,bins-2-1) 
    % If Chi2_lin_mark < chi2_lin_mark, hypothesis cannot be rejected 
    % (yes, probably normal!)

    
    
        % third order
figure; subplot(2,1,1)
plot(xvals2, yvals2,'mo'); hold on

xlabel('Year'); ylabel('pH anomaly');
[p4 e4] = polyfit(xvals2,yvals2',3);
[fitvals4 d2] = polyval(p4,xvals2,e4);
plot(xvals2,fitvals4,'b');
plot(xvals2,fitvals4-d2, 'r--')
plot(xvals2,fitvals4+d2, 'r--')
hold off
title('third-order fit for Regime 2')
legend('detrended pH', 'third-order fit')
hold off

subplot(2,1,2)
lin_res4 = yvals2'-fitvals4;
stem(xvals2, lin_res4, 'fill','--', 'MarkerFaceColor','r'); 
title('third-order linear regression residuals')
ylabel('pH anomaly')
xlabel('Year')

    % normality of residuals
figure
bins = 20;
%bins = round(sqrt(length(lin_res4)));
[lin1 centers] = hist(lin_res4,bins);
hist(lin_res4,bins)
xlabel('Residual')
ylabel('Occurrences')
title('Distribution of residuals for third-order fit')
created1 = pdf('norm',centers,mean(lin_res4),std(lin_res4)); 
created_new = created1.*(sum(lin1)./sum(created1));
lin_num = (lin1 - created_new).^2;
chi2_lin = sum(lin_num./created_new) 
chi2_lin_mark = chi2inv(0.95,bins-2-1) 
    % If Chi2_lin_mark < chi2_lin_mark, hypothesis cannot be rejected 
    % (yes, probably normal!)
    
[r1 p1 rlow1 rhigh1] = corrcoef(yvals2',fitvals4);

some2 = polyfit(yvals2',fitvals4,1); % slope = 0.5968
stuff2 = polyval(some2,yvals2');
figure;
plot(yvals2',fitvals4,'ro',yvals2',stuff2,'k-')

    % slope error
errvar2 = sum(((fitvals9-stuff2).^2)/(length(yvals2')-2));
slopestd2 = sqrt(errvar2./var(yvals2'));
tval2 = abs(tinv(0.025,length(yvals2')-2));
low2 = some2(1)-(tval2.*slopestd2)
high2 = some2(1)+(tval2.*slopestd2)
some2(1)
ylabel('fitted (model) values')
xlabel('data values')
title(sprintf('Slope is %d',some2(1)))
legend('Predicted values', 'Third-order fit')

%{
    % second order
figure; subplot(2,1,1)
plot(xvals2, yvals2,'mo'); hold on

xlabel('Year'); ylabel('pH anomaly');
p5 = polyfit(xvals2,yvals2',2);
fitvals5 = polyval(p5,xvals2);
title('second-order fit')
plot(xvals2,fitvals5,'b--'); hold off
legend('detrended pH', 'second-order fit')

subplot(2,1,2)
lin_res5 = yvals2'-fitvals5;
stem(xvals2, lin_res5, 'fill','--', 'MarkerFaceColor','r')
title('linear regression residuals')
ylabel('pH anomaly')

[r2 p2 rlow2 rhigh2] = corrcoef(yvals2',fitvals5)

    % third order

figure; subplot(2,1,1)
plot(xvals2, yvals2,'mo'); hold on

xlabel('Year'); ylabel('pH anomaly');
p6 = polyfit(xvals2,yvals2',3);
fitvals6 = polyval(p6,xvals2);
title('third-order fit')
plot(xvals2,fitvals6,'b--'); hold off
legend('detrended pH', 'third-order fit')

subplot(2,1,2)
lin_res6 = yvals2'-fitvals6;
stem(xvals2, lin_res6, 'fill','--', 'MarkerFaceColor','r')
title('linear regression residuals')
ylabel('pH anomaly')

[r2 p2 rlow2 rhigh2] = corrcoef(yvals2',fitvals6)

some2 = polyfit(yvals2',fitvals6,1)
%}




%     % reduced major axis
% b1 = std(yvals2')./std(xvals2);
% b0 = mean(yvals2')-(b1.*mean(xvals2));
% reduc_yvals = polyval([b1 b0],xvals2);
% plot(xvals2,reduc_yvals,'b--');
% 
%     % principle component
% means_out = [xvals2-mean(xvals2) yvals2'-mean(yvals2')];
% [u,s,v]=svd(means_out,'econ');
% v=v';
% xvals_pc = u(:,1)*s(1,1)*v(1,1)+mean(xvals2);
% yvals_pc = u(:,1)*s(1,1)*v(1,2)+mean(yvals2');
% coeffs_pc = polyfit(xvals_pc,yvals_pc,1);
% pc_yvals = polyval(coeffs_pc,xvals2);
% plot(xvals2,pc_yvals,'k--')

% legend('data','linear regression', 'rma', 'principle component')
% hold off







%% Gyre portion

cd('/Users/owner042/Dropbox/2015-Spring/Data-Analysis/Final')
load ('mdtm_data_1861_2100.mat')
data = mdtm_data;

xlim = [-136 -120]; ylim = [34 45]; % CCS region
[i,j]=rgrd_FindIJ(data.lon, data.lat, xlim, ylim);
ph_index = sq(meanNaN(meanNaN(data.pha_iso(i,j,:),1),2));
yvals22 = nn(ph_index);
figure;
plot(data.year, yvals22, 'm'); hold on
xlabel('Year'); ylabel('Anomalies from 1861-2005 mean');

cd('/Users/owner042/Dropbox/Acid-Namrata/3Isopycnals')
load ('Pacific_1861_2100.mat')
cd('/Users/owner042/Dropbox/2015-Spring/Data-Analysis/Final')
data2 = pac_alldata;

mfig; rnc_map(data2.pha_iso(:,:,1),data2)

xlim = [-190 -170]; ylim = [40 50]; % North Pacific gyre region
[i,j]=rgrd_FindIJ(data.lon, data.lat, xlim, ylim);

ph_index2 = sq(meanNaN(meanNaN(data.pha_iso(i,j,:),1),2));
yvals2 = nn(ph_index2);
xvals2 = data.year;
plot(xvals2, yvals2, 'g'); 
legend('pH anomalies in the CCS','pH anomalies in the gyre')
hold off

yvals7 = nn(ph_index2);
yvals7 = yvals7(1:120); % gyre region
yvals8 = yvals22(1:120); % CCS region
xvals7 = data.year;
xvals7 = xvals7(1:120);
figure;
plot(xvals7, yvals7, 'm'); hold on % gyre region
plot(xvals7, yvals8, 'b'); % CCS region
xlabel('Year'); ylabel('pH anomaly');
legend('pH anomalies in gyre', 'pH anomalies in CCS')
hold off

figure; 
plot(xvals7, detrend(yvals7), 'm'); hold on % gyre region
plot(xvals7, detrend(yvals8), 'b'); % CCS region
xlabel('Year'); ylabel('pH anomaly');
legend('pH anomalies in gyre', 'pH anomalies in CCS')
hold off

figure; 
yvas = detrend(yvals7) - detrend(yvals8); % neg #s mean gyre is leading CCS
plot(xvals7, yvas,'k*'); hold on
plot(xvals7, zeros(length(xvals7)),'r--'); hold off
