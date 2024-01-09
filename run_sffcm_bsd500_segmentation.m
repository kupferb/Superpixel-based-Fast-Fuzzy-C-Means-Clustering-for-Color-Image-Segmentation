clearvars
close all

if strcmp(getenv('computername'),'BENNYK')
    bsdsRoot = 'C:\DataSet\BSD\500\BSR_full\BSR\BSDS500\data';
    addpath C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels\others
    addpath C:\Users\Benny\MATLAB\Projects\Segmentation-Using-Superpixels\evals

else
    bsdsRoot = 'D:\DataSet\BSD\500\BSR_full\BSR\BSDS500\data';
    addpath D:\MATLAB\github\Segmentation-Using-Superpixels\others
    addpath D:\MATLAB\github\Segmentation-Using-Superpixels\evals

end
type = "test";
N_imgs = 200 * (type == "test") + 200 * (type == "train") + 100 * (type == "val");
ims_dir = fullfile(bsdsRoot,"images",type);
gt_dir = fullfile(bsdsRoot,"groundTruth",type);
ims_data = dir(fullfile(ims_dir,'*.jpg'));
ims_names = {ims_data.name}';
[~,sI] = sort(cell2mat(cellfun(@(x) str2double(x(1:end-4)),ims_names ,'uniformoutput',false)));
ims_names = ims_names(sI);
SE=3;
Nseg = 4;
[PRI_all,VoI_all,GCE_all,BDE_all] = deal(zeros(N_imgs,1));
for k_idxI = 1:length(ims_names)

    img_name = ims_names{k_idxI}(1:end-4);
    im = imread(fullfile(ims_dir, ims_names{k_idxI}));
    gt_segs_data = load(fullfile(gt_dir,sprintf('%s.mat',img_name)));
    num_seg = length(gt_segs_data.groundTruth );
    gt_imgs = cell(1,num_seg );
    for k_seg = 1:num_seg 
    
        gt_imgs{k_seg} = double(gt_segs_data.groundTruth{k_seg}.Segmentation );

    end
    
    L1=w_MMGR_WT(im,SE);
    L2=imdilate(L1,strel('square',2));
    [~,~,Num,centerLab]=Label_image(im,L2);
    %% fast FCM
    label_img=w_super_fcm(L2,centerLab,Num,Nseg);
%     Lseg=Label_image(im,Label);

    out_vals = eval_segmentation(label_img,gt_imgs);
    fprintf('(%d / %d) %6s: %2d %9.6f, %9.6f, %9.6f, %9.6f\n',k_idxI, length(ims_names), img_name, Nseg, out_vals.PRI, out_vals.VoI, out_vals.GCE, out_vals.BDE);
    
    PRI_all(k_idxI) = out_vals.PRI;
    VoI_all(k_idxI) = out_vals.VoI;
    GCE_all(k_idxI) = out_vals.GCE;
    BDE_all(k_idxI) = out_vals.BDE;



end
fprintf('Mean: %14.6f, %9.6f, %9.6f, %9.6f \n', mean(PRI_all), mean(VoI_all), mean(GCE_all), mean(BDE_all));
