
function set_ticks(x_tick, y_tick, z_tick, tick_unit)

    x_ticks = round(min(x_tick),1):(max(x_tick)-min(x_tick))/5:round(max(x_tick),1);
    set(gca,'xticklabels', round(x_ticks,1));
    y_ticks = round(min(y_tick),1):(max(y_tick)-min(y_tick))/5:round(max(y_tick),1);
    set(gca,'yticklabels', round(y_ticks,1));
    z_ticks = round(min(z_tick),1):(max(z_tick)-min(z_tick))/8:round(max(z_tick),1);
    set(gca,'zticklabels',round(z_ticks,1));
    
    xlabel(['x (', tick_unit, ')'], 'fontsize', 14);
    ylabel(['y (', tick_unit, ')'], 'fontsize', 14);
    zlabel(['z (', tick_unit, ')'], 'fontsize', 14);
end