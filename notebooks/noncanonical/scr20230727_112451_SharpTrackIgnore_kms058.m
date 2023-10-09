%% scr20230727_112451_SharpTrackIgnore_kms058.mlx
% 

addpath('\\ettina\Magill_Lab\Kouichi Nakamura\Analysis\allenCCFdata')
cd('\\ettina\Magill_Lab\Kouichi Nakamura\Analysis\Images from Otto\20230406 kms058') %TODO

subject_id = 'kms058' %TODO

% * remember to run one cell at a time, instead of the whole script at once *

% directory of histology images
image_folder = '\\ettina\Magill_Lab\Kouichi Nakamura\Analysis\Images from Otto\20230406 kms058\RGB_ignore';  %TODO


% directory to save the processed images -- can be the same as the above image_folder
% results will be put inside a new folder called 'processed' inside of this image_folder
save_folder = image_folder;

% name of images, in order anterior to posterior or vice versa
% once these are downsampled they will be named ['original name' '_processed.tif']
image_file_names = dirregexp(image_folder, '.+tif$'); % get the contents of the image_folder
image_file_names = natsortfiles({image_file_names.name});

%% 
% 


disp(image_file_names')
% image_file_names = {'slide no 2_RGB.tif','slide no 3_RGB.tif','slide no 4_RGB.tif'}; % alternatively, list each image in order

% if the images are individual slices (as opposed to images of multiple
% slices, which must be cropped using the cell CROP AND SAVE SLICES)
image_files_are_individual_slices = true;

% use images that are already at reference atlas resolution (here, 10um/pixel)
use_already_downsampled_image = false; %TODO

% pixel size parameters: microns_per_pixel of large images in the image
% folder (if use_already_downsampled_images is set to false);
% microns_per_pixel_after_downsampling should typically be set to 10 to match the atlas
microns_per_pixel = 2.6 *3 ; %TODO 
microns_per_pixel_after_downsampling = 10 ;%TODO



% ----------------------
% additional parameters
% ----------------------

% if the images are cropped (image_file_are_individual_slices = false),
% name to save cropped slices as; e.g. the third cropped slice from the 2nd
% image containing many slices will be saved as: save_folder/processed/save_file_name02_003.tif
save_file_name = [subject_id,'_'];

% increase gain if for some reason the images are not bright enough
gain = 1; 

% plane to view ('coronal', 'sagittal', 'transverse')
plane = 'sagittal'; %TODO

% size in pixels of reference atlas brain. For coronal slice, this is 800 x 1140
if strcmp(plane,'coronal')
    atlas_reference_size = [800 1140]; 
elseif strcmp(plane,'sagittal')
    atlas_reference_size = [800 1320]; 
elseif strcmp(plane,'transverse')
    atlas_reference_size = [1140 1320];
end



% finds or creates a folder location for processed images -- 
% a folder within save_folder called processed
folder_processed_images = fullfile(save_folder, 'processed');
if ~exist(folder_processed_images)
    mkdir(folder_processed_images)
end
%% 2. LOAD AND PROCESS SLICE PLATE IMAGES (mandatory)
% 
% 
% apparently *only works with 8 bit RGB images*
% 
% Only the first channel was visible when used multichannels in 16 bit
% 
% 
%%
% 
%   % close all figures
%   close all
%      
%   
%   % if the images need to be downsampled to 10um pixels (use_already_downsampled_image = false), 
%   % this will downsample and allow you to adjust contrast of each channel of each image from image_file_names
%   %
%   % if the images are already downsampled (use_already_downsampled_image = true), this will allow
%   % you to adjust the contrast of each channel
%   %
% Open Histology Viewer figure
try; figure(histology_figure);
catch; histology_figure = figure('Name','Histology Viewer'); end

warning('off', 'images:initSize:adjustingMag'); warning('off', 'MATLAB:colon:nonIntegerIndex');

% Function to downsample and adjust histology image
HistologyBrowser(histology_figure, save_folder, image_folder, image_file_names, folder_processed_images, image_files_are_individual_slices, ...
    use_already_downsampled_image, microns_per_pixel, microns_per_pixel_after_downsampling, gain)
%
%% 
% Controls: 
% 
% 
% 
% space: adjust contrast for current channel / return to image-viewing mode 
% 
% e: view original version 
% 
% any key: return to modified version 
% 
% r: reset to original 
% 
% c: move to next channel 
% 
% s: save image 
% 
% left/right arrow: save and move to next slide image 
% 
% 
% 
% Space
% 
% Drag histogram or type values
% 
% Adjust Data
% 
% Space on the main figure to go back to the default mode
% 3. GO THROUGH TO FLIP HORIZONTAL SLICE ORIENTATION, ROTATE, SHARPEN, and CHANGE ORDER (mandatory)
% 
% 
% 
%%
% 
%   
%   % close all figures
%   close all
%               
%   % this takes images from folder_processed_images ([save_folder/processed]),
%   % and allows you to rotate, flip, sharpen, crop, and switch their order, so they
%   % are in anterior->posterior or posterior->anterior order, and aesthetically pleasing
%   % 
%   % it also pads images smaller than the reference_size and requests that you
%   % crop images larger than this size
%   %
%   % note -- presssing left or right arrow saves the modified image, so be
%   % sure to do this even after modifying the last slice in the folder
slice_figure = figure('Name','Slice Viewer');
SliceFlipper(slice_figure, folder_processed_images, atlas_reference_size)

%% 
% Controls: 
% 
% 
% 
% right: save and see next image 
% 
% left: save and see previous image 
% 
% scroll: rotate slice 
% 
% s: sharpen 
% 
% g: toggle grid 
% 
% c: crop slice further 
% 
% f: flip horizontally 
% 
% w: switch order (move image forward) 
% 
% r: reset to original 
% 
% delete: delete current image 
%% 4. Navigating in the reference atlas (|\Navigate_Atlas_and_Register_Slices.m|)
% 
% ENTER FILE LOCATION AND PROBE-SAVE-NAME



% directory of histology
processed_images_folder = folder_processed_images; %TODO

% name the saved probe points, to avoid overwriting another set of probes going in the same folder
probe_save_name_suffix = '_probe'; %TODO

% directory of reference atlas files
annotation_volume_location = "\\ettina\Magill_lab\Kouichi Nakamura\Analysis\allenCCFdata\annotation_volume_10um_by_index.npy";
structure_tree_location = "\\ettina\Magill_lab\Kouichi Nakamura\Analysis\allenCCFdata\structure_tree_safe_2017.csv";
template_volume_location = "\\ettina\Magill_lab\Kouichi Nakamura\Analysis\allenCCFdata\template_volume_10um.npy";

% plane to view ('coronal', 'sagittal', 'transverse')
% plane = 'sagittal';
% 5.  GET PROBE TRAJECTORY POINTS
% 


% load the reference brain and region annotations
if ~exist('av','var') || ~exist('st','var') || ~exist('tv','var')
    disp('loading reference atlas...')
    av = readNPY(annotation_volume_location);
    st = loadStructureTree(structure_tree_location);
    tv = readNPY(template_volume_location);
end

% select the plane for the viewer
if strcmp(plane,'coronal')
    av_plot = av;
    tv_plot = tv;
elseif strcmp(plane,'sagittal')
    av_plot = permute(av,[3 2 1]);
    tv_plot = permute(tv,[3 2 1]);
elseif strcmp(plane,'transverse')
    av_plot = permute(av,[2 3 1]);
    tv_plot = permute(tv,[2 3 1]);
end
%% 6.  Slice Viewer and Atlas Viewer
% 
% 
% %TODO Changing a slice on Slice Viewer is not reflected on Atlas Viewer. The 
% two views are not synchronized.
% 
% %TODO When moving to a new slice in Slice Viewer and add points to both Atlas 
% and Slice Viewers, the number of slice points is over counted.
% 
% 
% 
% 
% 
% |Controls:| 
% 
% |---------| 
% 
% |Navigation:| 
% 
% *Switch scroll modes*
% 
% |up: scroll through A/P angles (for coronal sections) *(D/V for sagittal)*|
% 
% |right: scroll through M/L angles  (for coronal sections) *(A/P for sagittal)*|
% 
% |down: scroll through *atlas* slices (%TODO bug? *switch scroll mode -- scroll 
% along M/L axis*)|
% 
% |*left: scroll the matched slices (Slice 1, Slice 2, ...)*|
% 
% 
% 
% |scroll: move between slices *or angles*|
% 
% 
% 
% |Registration:| 
% 
% |t: toggle mode where clicks are logged for transform| 
% 
% |h: toggle overlay of current histology slice| 
% 
% |p: toggle mode where clicks are logged for probe or switch probes| 
% 
% |n: add a new probe| 
% 
% |x: save transform and current atlas location| 
% 
% |l: load transform for current slice; press again to load probe points| 
% 
% |s: save current probe| 
% 
% |d: delete most recent probe point or transform point| 
% 
% |w: enable/disable probe viewer mode for current probe|  
% 
% 
% 
% |Viewing modes:| 
% 
% |o: toggle overlay of current region extent| 
% 
% |a: toggle to viewing boundaries| 
% 
% |v: toggle to color atlas mode| 
% 
% |g: toggle gridlines| 
% 
% 
% 
% |space: display controls| 
% 
% 
%%
% 
% create Atlas viewer figure
f1 = figure('Name','Atlas Viewer');

% show histology in Slice Viewer
try; figure(slice_figure_browser); title('');
catch; slice_figure_browser = figure('Name','Slice Viewer'); end
f2 = slice_figure_browser

reference_size = size(tv_plot);
sliceBrowser(slice_figure_browser, processed_images_folder, f1, reference_size);

% % use application in Atlas Transform Viewer
% % use this function if you have a processed_images_folder with appropriately processed .tif histology images
AtlasTransformBrowser(f1, tv_plot, av_plot, st, slice_figure_browser, processed_images_folder, probe_save_name_suffix, plane);
%   
%   
%   
%   % use the simpler version, which does not interface with processed slice images
%   % just run these two lines instead of the previous 5 lines of code
%   % 
%   %  save_location = processed_images_folder;
%   %  f = allenAtlasBrowser(f, tv_plot, av_plot, st, save_location, probe_save_name_suffix, plane);
%   
%   
%
%% |8.| Display Probe Track
% 


%% ENTER PARAMETERS AND FILE LOCATION

% file location of probe points
% processed_images_folder = 'C:\Drive\Histology\brainX\processed';

% directory of reference atlas files
% annotation_volume_location = 'C:\Drive\Histology\for tutorial\annotation_volume_10um_by_index.npy';
% structure_tree_location = 'C:\Drive\Histology\for tutorial\structure_tree_safe_2017.csv';

% name of the saved probe points
% probe_save_name_suffix = 'electrode_track_1';
% probe_save_name_suffix = '';

% either set to 'all' or a list of indices from the clicked probes in this file, e.g. [2,3]
probes_to_analyze = 'all';  % [1 2]

% --------------
% key parameters
% --------------
% how far into the brain did you go from the surface, either for each probe or just one number for all -- in mm

% depth from recording notes: This won't to be usef for plotting or computation
probe_lengths = [...
    1.9000, ... OF
    5.5060, ...
    5.0943, ...
    5.4270, ...
    4.6550, ...
    5.3120, ...
    5.0493, ...
    5.3120, ...
    4.3810, ...
    ]

% from the bottom tip, how much of the probe contained recording sites -- in mm
active_probe_length = 3.84;

% distance queried for confidence metric -- in um
probe_radius = 32; 

% overlay the distance between parent regions in gray (this takes a while)
% show_parent_category = false; 

% plot this far or to the bottom of the brain, whichever is shorter -- in mm
distance_past_tip_to_plot = .5;

% set scaling e.g. based on lining up the ephys with the atlas
% set to *false* to get scaling automatically from the clicked points
% scaling_factor = false;


% ---------------------
% additional parameters
% ---------------------
% plane used to view when points were clicked ('coronal' -- most common, 'sagittal', 'transverse')
% plane = 'coronal';

% probe insertion direction 'down' (i.e. from the dorsal surface, downward -- most common!) 
% or 'up' (from a ventral surface, upward)
% probe_insertion_direction = 'down';

% show a table of regions that the probe goes through, in the console
% show_region_table = true;
% 
% % black brain?
% black_brain = true;


% close all
%% 9 and 10. Plot and compute

[probes, fwireframe, fig_probes, T_probes, Tapdvml_contacts, T_borders] ...
    = plot_and_compute_probe_positions_from2(av, st, subject_id, plane,...
    processed_images_folder, probe_save_name_suffix, probes_to_analyze, probe_lengths,...
    active_probe_length, probe_radius)
%%
fwireframe.Visible = 'on';
savefig(fwireframe,'fwireframe_ignore.fig')
fwireframe.Visible = 'off';

%% 11. Matching with the sessions
% We need to identify which probes are for which session based on the color 
% of the dye and positions in pictures.
% 
% Probably, you need to use *Atlas Viewers* again and compare them with OMERO 
% and descriptions in the experimental notebook. Hit *L* key twice to load probe 
% points.
% 
% 
% 
% <http://julien-pc.mrc.ox.ac.uk:4080/webclient/?show=dataset-268 http://julien-pc.mrc.ox.ac.uk:4080/webclient/?show=dataset-268>
% 
% 
% 
% Add columns to each table
% 
% *probe_id* : intger
% 
% *probe_AB* : 'A'  | 'B' | 'optic_fiber' 
% 
% *session_id*: 'kms058-2023-03-24-151254'
% 
% *subject_id*: 'kms058'
% 
% 

struct_probes = preallocatestruct(...
    {'probe_id', 'probe_AB', 'session_id', 'subject_id', 'probe_note', 'upward_from_tip_um'},...
    size(probes)');

for i = 1:length(probes)
    struct_probes(i).probe_id = i;
    struct_probes(i).probe_AB = '';
    struct_probes(i).session_id = '';
    struct_probes(i).subject_id = subject_id;
    struct_probes(i).probe_note = '';
    struct_probes(i).upward_from_tip_um = 0;
end

% white
struct_probes(1).probe_AB = 'optic fiber';

% gold, DiI and Neuro-DiO for CPu/GPe
struct_probes(2).probe_AB = 'A'; 
struct_probes(2).session_id = 'kms058-2023-03-24-151254'; 

% turquoise, DiD and Neuro-DiO for CPu/GPe
struct_probes(3).probe_AB = 'A';
struct_probes(3).session_id = 'kms058-2023-03-25-184034';

% fern, Neuro-DiO for CPu/GPe
struct_probes(4).probe_AB = 'A';
struct_probes(4).session_id = 'kms058-2023-03-23-191740';


% bubble gum, DiI and Neuro-DiO for SNr
struct_probes(5).probe_AB = 'B';
struct_probes(5).session_id = 'kms058-2023-03-24-151254';
struct_probes(5).probe_note = 'Deepest point reached was 4655 µm but we moved up by 100 µm for recording to 4555 µm';
struct_probes(5).upward_from_tim_um = 100; 

% overcast sky, Neuro-DiO for SNr
struct_probes(6).probe_AB = 'B';
struct_probes(6).session_id = 'kms058-2023-03-23-191740';

% rawhide, DiD and Neuro-DiO for SNr
struct_probes(7).probe_AB = 'B';
struct_probes(7).session_id = 'kms058-2023-03-25-184034';

% green apple, DiI for SNr
struct_probes(8).probe_AB = 'B';
struct_probes(8).session_id = 'kms058-2023-03-20-132658';

% red, DiD for SNr
struct_probes(9).probe_AB = 'B';
struct_probes(9).session_id = 'kms058-2023-03-14-165110';

T_probes = struct2table(struct_probes);
T_probes.probe_AB = string(T_probes.probe_AB);
T_probes.session_id = string(T_probes.session_id);
T_probes.subject_id = string(T_probes.subject_id);
T_probes.probe_note = string(T_probes.probe_note);


%%
Tapdvml_contacts = prep_Tapdvml_contacts(T_probes, Tapdvml_contacts)

%%
writetable(T_borders, "borders_table.xlsx",'FileType','spreadsheet')
writetable(T_probes, "T_probes.xlsx",'FileType','spreadsheet')
writetable(Tapdvml_contacts, "Tapdvml_contacts.xlsx",'FileType','spreadsheet')
% 12. Export the summary figure
% 


FIG = setupFigure('Probe targets',[0 0 29.7 21]);
FIG.PaperOrientation = 'landscape'; % Add this line

FIG.Color = ones(1,3)*0.85


tl = tiledlayout(FIG, 1,length(fig_probes));
title(tl, subject_id)

for i = 1:length(fig_probes)
    ax = nexttile(tl, i);
    AX = findobj(fig_probes(i),'Type','Axes');

    yyaxis(AX,'left')
    yyaxis(ax,'left')

    copyobj( AX.Children, ax);
    copyobj(findall(AX, 'Type','patch'), ax);
    copygraphicprops(AX, ax, ["YTick","YTickLabel", "YLim",'YColor','YDir'])
    copygraphicprops(AX.XLabel, ax.XLabel, ["String"])
    copygraphicprops(AX.YLabel, ax.YLabel, ["String"])
    copygraphicprops(AX.Title,  ax.Title, ["String",'Color'])

    yyaxis(AX,'right')
    yyaxis(ax,'right')

    copyobj( AX.Children, ax);
    copygraphicprops(AX, ax, ["XLim","YTick","YTickLabel", "YLim",'YColor','YDir'])


end

%%

print(FIG,'-vector', 'all_probes','-dpdf')
print(FIG, '-r400','all_probes','-dpng')